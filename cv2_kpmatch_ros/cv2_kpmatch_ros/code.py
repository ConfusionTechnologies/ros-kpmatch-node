from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from colorsys import hsv_to_rgb
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point
from nicepynode import Job
from nicepynode.utils import (
    RT_PUB_PROFILE,
    RT_SUB_PROFILE,
    declare_parameters_from_dataclass,
    letterbox,
)
from rclpy.node import Node
from ros2topic.api import get_msg_class
from sensor_msgs.msg import Image
from visualization_msgs.msg import ImageMarker

from .cfg import kpDetCfg
from .utils import calc_features, filter_matches, find_object

NODE_NAME = "keypoint_detector"
cv_bridge = CvBridge()


@dataclass
class KpDetector(Job[kpDetCfg]):

    ini_cfg: kpDetCfg = field(default_factory=kpDetCfg)

    def attach_params(self, node: Node, cfg: kpDetCfg):
        super(KpDetector, self).attach_params(node, cfg)

        declare_parameters_from_dataclass(node, cfg)

    def attach_behaviour(self, node: Node, cfg: kpDetCfg):
        super(KpDetector, self).attach_behaviour(node, cfg)

        self._init_stuff(cfg)

        self.log.info(f"Waiting for publisher@{cfg.frames_in_topic}...")
        self.msg_type = get_msg_class(node, cfg.frames_in_topic, blocking=True)
        self._frame_sub = node.create_subscription(
            self.msg_type, cfg.frames_in_topic, self._on_input, RT_SUB_PROFILE
        )
        self._rect_pub = node.create_publisher(
            ImageMarkerArray, cfg.rect_out_topic, RT_PUB_PROFILE
        )
        self._markers_pub = node.create_publisher(
            ImageMarkerArray, cfg.kp_out_topic, RT_PUB_PROFILE
        )

        self._id_color_map = defaultdict(lambda: random.random())
        """Used to assign color for visualizing lmao"""

        self.log.info("Ready")

    def detach_behaviour(self, node: Node):
        super().detach_behaviour(node)

        node.destroy_subscription(self._frame_sub)
        node.destroy_publisher(self._rect_pub)
        node.destroy_publisher(self._markers_pub)

    def on_params_change(self, node: Node, changes: dict):
        self.log.info(f"Config changed: {changes}.")
        if "use_ocl" in changes:
            cv2.ocl.setUseOpenCL(changes["use_ocl"])
        if not all(
            n
            in (
                "min_matches",
                "use_flann",
                "ratio_thres",
                "homography_method",
                "inlier_thres",
                "use_bg_subtraction",
                "use_ocl",
                "debug",
            )
            for n in changes
        ):
            self.log.info(f"Config change requires restart.")
            return True
        return False

    def _init_stuff(self, cfg: kpDetCfg):
        if self.cfg.use_cuda:
            self.descriptor = self.detector = cv2.cuda.ORB_create(
                **cfg.test_detector.__dict__
            )

            bgsegm = cv2.cuda.createBackgroundSubtractorMOG2()
            self.bg_subtractor = lambda img: bgsegm.apply(
                img, learningRate=-1, stream=cv2.cuda.Stream.Null()
            )

            # no flann matcher, so make them both the same
            self.flann_matcher = (
                self.matcher
            ) = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        else:
            self.detector = cv2.ORB_create(**cfg.test_detector.__dict__)
            self.descriptor = cv2.xfeatures2d.BEBLID_create(1.0)
            bgsegm = cv2.bgsegm.createBackgroundSubtractorCNT(
                minPixelStability=int(cfg.rate),
                maxPixelStability=int(60 * cfg.rate),
                isParallel=True,  # comes at the cost of other processes ofc
            )
            self.bg_subtractor = lambda img: bgsegm.apply(img)

            # NOTE: no documentation exists for flann. i cannot figure out what
            # parameters exist or do... values here are hardcoded from tutorial
            # checks affect recursion level of flann to increase accuracy
            # ...but make it insanely large or 0 seems to do nothing
            # likely because it is the "upper-limit" & hence only affects
            # if the target is hidden/not present
            search_params = dict(checks=50)
            # SURF, SIFT, etc
            # FLANN_INDEX_KDTREE = 1
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # binary descriptors such as ORB, BRISK, BEBLID etc
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,  # 12
                key_size=12,  # 20
                multi_probe_level=1,  # 2
            )
            self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
            # dont use builtin crossCheck, its significantly slower than ratio test
            # supposedly, there was an OCL version of BFMatcher, but posts online say it was bugged
            # cant figure out how to get it work in 4.6.0 anyways.
            self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        self._calc_features(cfg)

    def _calc_features(self, cfg: kpDetCfg):
        with open(cfg.database_path, "r") as f:
            db: dict = json.load(f)

        self.features = {}
        query_detector = (cv2.cuda.ORB_create if self.cfg.use_cuda else cv2.ORB_create)(
            **cfg.query_detector.__dict__
        )
        for name, path in db.items():
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert not im is None, f"Failed to read image {name} from {path}"
            if cfg.scale_wh:
                size = (
                    (cfg.scale_wh, cfg.scale_wh)
                    if isinstance(cfg.scale_wh, int)
                    else cfg.scale_wh
                )

                im = letterbox(im, size, color=(0, 0, 0))[0]
            shape = im.shape
            if self.cfg.use_cuda:
                buf = cv2.cuda_GpuMat()
                buf.upload(im)
                im = buf
            self.features[name] = calc_features(
                im, query_detector, self.descriptor, use_cuda=self.cfg.use_cuda
            ) + (shape[0], shape[1])

    def _detect(self, img: np.ndarray):
        # TODO: write keypoint tracker
        # https://docs.opencv.org/4.x/d5/dec/classcv_1_1videostab_1_1KeypointBasedMotionEstimator.html
        # will reduce lag if no need to match every frame
        # meanshift will also help
        # worst case... one process per query image?
        shape = img.shape[:2]
        if self.cfg.use_cuda:
            buf = cv2.cuda_GpuMat()
            buf.upload(img)
            img = buf
        elif self.cfg.use_ocl:
            img = cv2.UMat(img)
        mask = self.bg_subtractor(img) if self.cfg.use_bg_subtraction else None
        img = (cv2.cuda.cvtColor if self.cfg.use_cuda else cv2.cvtColor)(
            img, cv2.COLOR_BGR2GRAY
        )
        t_kp, t_desc = calc_features(
            img, self.detector, self.descriptor, mask, use_cuda=self.cfg.use_cuda
        )

        o = {}
        results = []
        o["dets"] = results
        if not self.cfg.use_cuda and len(t_kp) == 0:
            return o

        # standardize box coords
        wh = shape[::-1]

        matched_kp = []
        for name, (q_kp, q_desc, qh, qw) in self.features.items():
            matcher = self.flann_matcher if self.cfg.use_flann else self.matcher
            if self.cfg.use_cuda:
                pairs = matcher.knnMatchAsync(
                    q_desc, t_desc, k=2
                )  # pairs of (best, 2nd best) matches
            else:
                pairs = matcher.knnMatch(
                    q_desc, t_desc, k=2
                )  # pairs of (best, 2nd best) matches
            matches = filter_matches(pairs, ratio_thres=self.cfg.ratio_thres)
            if len(matches) < self.cfg.min_matches:
                continue
            transform, _ = find_object(
                matches,
                q_kp,
                t_kp,
                method=self.cfg.homography_method,
                inlier_thres=self.cfg.inlier_thres,
            )
            # tl, bl, br, tr
            box = np.float32(
                ((0, 0), (0, qh - 1), (qw - 1, qh - 1), (qw - 1, 0))
            ).reshape(-1, 1, 2)
            try:
                box = cv2.perspectiveTransform(box, transform).reshape(-1, 2)
                # box /= wh
                results.append((name, box))
                if self.cfg.debug:
                    matched_kp.append(
                        (
                            name,
                            np.array(
                                tuple(t_kp[m.trainIdx].pt for m in matches)
                            ).reshape(-1, 2)
                            # / wh,
                        )
                    )

            except:
                pass
        if self.cfg.debug:
            o["debug"] = {
                "all_kp": cv2.KeyPoint_convert(t_kp),  # / wh,
                "matched_kp": matched_kp,
            }
        return o

    def _on_input(self, msg: Image):
        if (
            self._rect_pub.get_subscription_count()
            + self._markers_pub.get_subscription_count()
            < 1
        ):
            return

        if isinstance(msg, Image):
            img = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        else:
            img = cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if 0 in img.shape:
            self.log.debug("Image has invalid shape!")
            return

        result = self._detect(img)

        if self._rect_pub.get_subscription_count() > 0:
            rectsmsg = ImageMarkerArray()

            for name, box in result["dets"]:
                color = hsv_to_rgb(self._id_color_map[name], 1, 1)
                rect = ImageMarker(header=msg.header)
                rect.scale = 4.0
                rect.type = ImageMarker.POLYGON
                rect.outline_color.r = float(color[0])
                rect.outline_color.g = float(color[1])
                rect.outline_color.b = float(color[2])
                rect.outline_color.a = 1.0

                for p in box:
                    rect.points.append(Point(x=p[0] / 1.0, y=p[1] / 1.0))

                rectsmsg.markers.append(rect)

            self._rect_pub.publish(rectsmsg)

        if (
            self.cfg.debug
            and "debug" in result
            and self._markers_pub.get_subscription_count() > 0
        ):
            markersmsg = ImageMarkerArray()

            all_pts = ImageMarker(header=msg.header)
            all_pts.scale = 2.0
            all_pts.type = ImageMarker.POINTS
            all_pts.outline_color.r = 1.0
            all_pts.outline_color.a = 1.0

            for p in result["debug"]["all_kp"]:
                all_pts.points.append(Point(x=p[0] / 1.0, y=p[1] / 1.0))

            markersmsg.markers.append(all_pts)

            for name, kps in result["debug"]["matched_kp"]:
                color = hsv_to_rgb(self._id_color_map[name], 1, 1)
                matched_pts = ImageMarker(header=msg.header)
                matched_pts.scale = 4.0
                matched_pts.type = ImageMarker.POINTS
                matched_pts.outline_color.r = float(color[0])
                matched_pts.outline_color.g = float(color[1])
                matched_pts.outline_color.b = float(color[2])
                matched_pts.outline_color.a = 1.0

                for p in kps:
                    matched_pts.points.append(Point(x=p[0] / 1.0, y=p[1] / 1.0))

                markersmsg.markers.append(matched_pts)

            self._markers_pub.publish(markersmsg)


def main(args=None):
    if __name__ == "__main__" and args is None:
        args = sys.argv

    try:
        rclpy.init(args=args)

        node = Node(NODE_NAME)

        cfg = kpDetCfg()
        KpDetector(node, cfg)

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
