import cv2


def calc_features(img, detector, descriptor=None, mask=None, keypoints=None):
    """Calculate (keypoints, descriptors, height, width) given an image.

    Args:
        detector (Any): Keypoint detector to use.
        img (np.ndarray): BGR image in HWC order as uint8.
        mask (np.ndarray): bitmask where 1 indicates region of interests.
        keypoints (_type_, optional): Manually specified keypoints to use. Defaults to None.
    """
    assert len(img.shape) == 2, "Image must be grayscale!"
    kps = keypoints if keypoints else detector.detect(img, mask)
    kps, desc = (
        descriptor.compute(img, kps) if descriptor else detector.compute(img, kps)
    )
    return kps, desc, img.shape[0], img.shape[1]


# best explanation of homography method I could find:
# http://amroamroamro.github.io/mexopencv/matlab/cv.findHomography.html
# Given very few outliers (according to test), least squares is best
# In brief:
# Least Squares (0): Use all points. Effective only when few outliers
# cv2.RANSAC & cv2.RHO:
#   - attempts to find best set of inliers, but needs threshold
#   - RHO is more accurate but needs more points than RANSAC
# cv2.LMEDS: like voting, needs at least 50% inliers
def find_object(matches, query_kp, test_kp, method=0, inlier_thres=5.0):
    # coordinates in query & test image that match
    query_pts = cv2.KeyPoint_convert(
        query_kp, tuple(m.queryIdx for m in matches)
    ).reshape(-1, 1, 2)
    test_pts = cv2.KeyPoint_convert(
        test_kp, tuple(m.trainIdx for m in matches)
    ).reshape(-1, 1, 2)
    transform, mask = cv2.findHomography(query_pts, test_pts, method, inlier_thres)
    return transform, mask


# @cython.compile
def filter_matches(pairs, ratio_thres=0.75):
    """Use Lowe's ratio test to filter pairs of matches obtained from knnMatch"""
    # pair is (best, 2nd best), check if best is closer by factor compared to 2nd best
    return [
        p[0]
        for p in pairs
        if len(p) == 1 or (len(p) == 2 and p[0].distance < ratio_thres * p[1].distance)
    ]
