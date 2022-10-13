from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import cv2
from nicepynode import JobCfg


@dataclass
class orbCfg:
    """OpenCV default settings. See https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html"""

    nfeatures: int = 500
    """max limit on number of features to be detected"""
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31
    fastThreshold: int = 20


@dataclass
class queryDetCfg(orbCfg):
    nfeatures: int = 1000
    scaleFactor: float = 1.2
    nlevels: int = 4
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31
    fastThreshold: int = 20


@dataclass
class testDetCfg(orbCfg):
    """to minimize lag, test detector uses only one scale to extract features,
    expecting the query detector to have detected features at several scales beforehand.
    """

    nfeatures: int = 1000
    scaleFactor: float = 1.0
    nlevels: int = 1
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31
    fastThreshold: int = 20


@dataclass
class kpDetCfg(JobCfg):
    query_detector: queryDetCfg = field(default_factory=queryDetCfg)
    """Config for detector used to extract features from images"""
    test_detector: testDetCfg = field(default_factory=testDetCfg)
    """Config for detector used to extract features from livefeed"""
    min_matches: int = 40
    """min keypoint matches to consider a detection"""
    use_flann: bool = True
    """use flann-based matcher, tested to be significantly lightweight for large number of features,  
    but buggy openCV implementation means when nothing is detected, it will give many (filterable) false positives.
    However, it is single-threaded unlike brute force, so throughput will drop (it is still much more efficient).
    """
    use_cuda: bool = False
    """use CUDA-accelerated variants of everything. 
    Is broken for the foreseeable future."""
    use_ocl: bool = False
    """use openCV transparent openCL API for possible performance boost.
    Under stress test conditions (>1m features, no bg removal, 16 levels), had performance penalty, so likely useless."""
    scale_wh: Union[int, tuple[int, int]] = 480
    """scale props to fit this resolution"""
    normalize_coords: bool = False
    """Whether to normalize coords."""
    ratio_thres: float = 0.8
    """threshold for keypoint to be considered a match"""
    homography_method: int = cv2.RANSAC
    """See http://amroamroamro.github.io/mexopencv/matlab/cv.findHomography.html"""
    inlier_thres: float = 5.0
    """threshold for homography_method"""
    use_bg_subtraction: bool = True
    """use background subtraction, significant performance improvement if image is 
    feature-rich & keypoint detectors are set to go nuts. Else slight performance penalty."""
    database_path: str = "/data/props.json"
    """Path to json file specifying map of prop name to image path."""
    frames_in_topic: str = "~/frames_in"
    """Topic to receive frames from."""
    rect_out_topic: str = "~/rect_out"
    """Rectangle output that might be skewed and rotated."""
    kp_out_topic: str = "~/kp_out"
    """Keypoints for visualization & debug."""
    debug: bool = True
    """Whether to generate & send debug points."""
