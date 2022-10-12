# ros-kpmatch-node

Image feature keypoint matching

## Notes

- OpenCV implementation of feature extractors are multi-threaded. Using absurd values will cause the entire system to lag when the image is feature rich.
- TODO: implement multiple objects. Even if we dont have duplicate objects, it provides a performance boost to match objects cluster-wise instead of matching the whole image.
  - Cluster keypoints using meanshift, then match each cluster (method found online)
  - See:
    - <https://stackoverflow.com/questions/52425355/how-to-detect-multiple-objects-with-opencv-in-c>
    - <https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects>
    - <https://stackoverflow.com/questions/17357190/using-surf-descriptors-to-detect-multiple-instances-of-an-object-in-opencv>
      - ROI Sweep, not as stupid as my naive method (i saw one stackoverflow using my method lmao) (still limited by scale, sounds like YOLO)
    - <https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py>
      - expert confirmation meanshift is the "correct" approach: <https://answers.opencv.org/question/17985/detecting-multiple-instances-of-same-object-with-keypoint-matching-approach/>
- TBH if ppl were still researching keypoint detection, there might be damn good methods by now methods that can match neural networks with much less data (like a better version of HAAR) but no it got abandoned in favour of neural networks
- NOTE: Usage of ORB Detector & BEBLID Descriptor is hardcoded, trying to make it configurable is difficult due to different parameters involved...
- NOTE: GPU Acceleration doesn't seem feasible
  - Either openCV's openCL-based transparent API doesn't cover the functions we are using, or the overhead of GPU transfer dominates CPU usage
  - CUDA requires significant messing up of everything to work, and likely will also have the overhead issue
  