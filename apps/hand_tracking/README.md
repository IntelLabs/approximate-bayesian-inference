
### Installation

####Install pyrealsense2

####Install ar_marker dependency

```shell
git clone https://github.com/DebVortex/python-ar-markers.git
cd python-ar-markers
```

Fix an implicit integer cast to work with python3
```
diff --git a/ar_markers/hamming/detect.py b/ar_markers/hamming/detect.py
index 6ddc963..eea1c20 100644
--- a/ar_markers/hamming/detect.py
+++ b/ar_markers/hamming/detect.py
@@ -71,7 +71,7 @@ def detect_markers(img):
 
         _, warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
         marker = warped_bin.reshape(
-            [MARKER_SIZE, warped_size / MARKER_SIZE, MARKER_SIZE, warped_size / MARKER_SIZE]
+            [MARKER_SIZE, int(warped_size / MARKER_SIZE), MARKER_SIZE, int(warped_size / MARKER_SIZE)]
         )
```

Install the package in your user space
```shell
python3 setup.py install --user
```
