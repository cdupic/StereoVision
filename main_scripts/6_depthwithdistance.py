import cv2
import numpy as np
import torch
import os
import json
from start_cameras import Start_Cameras

# Globals
sbm = None
disparity_normalized = None

# Load YOLOv5 model (CPU)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
model.conf = 0.5  # detection threshold

def load_map_settings(file):
    global sbm
    if not os.path.isfile(file):
        print("⚠️ No settings file found, using defaults.")
        SWS, PFS, PFC = 15, 5, 31
        MDS, NOD = 0, 96
        TTH, UR, SR, SPWS = 100, 10, 15, 100
    else:
        with open(file, 'r') as f:
            data = json.load(f)
        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

    sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

def stereo_depth_map(left, right):
    global disparity_normalized
    left = cv2.GaussianBlur(left, (5, 5), 0)
    right = cv2.GaussianBlur(right, (5, 5), 0)
    disparity = sbm.compute(left, right)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_filtered = cv2.medianBlur(np.uint8(disparity_normalized), 5)
    disparity_color = cv2.applyColorMap(disparity_filtered, cv2.COLORMAP_TURBO)
    return disparity_color

def estimate_distance(x, y, size=5):
    if disparity_normalized is None:
        return None
    h, w = disparity_normalized.shape
    x1, x2 = max(0, x - size), min(w, x + size)
    y1, y2 = max(0, y - size), min(h, y + size)
    roi = disparity_normalized[y1:y2, x1:x2]
    valid = roi[roi > 0]
    return 1000 / np.median(valid) if valid.size > 0 else None


def detect_and_annotate(frame):
    results = model(frame)
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # person
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = estimate_distance(cx, cy)
            label = f"Person {dist:.1f} cm" if dist else "Person ???"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

if __name__ == "__main__":
    cam = Start_Cameras(0).start()
    load_map_settings("../3dmap_set.txt")
    cv2.namedWindow("DepthMap")

    while True:
        grabbed, frame = cam.read()
        if not grabbed:
            continue

        # Simule deux vues : image originale + décalée
        left_frame = frame.copy()
        right_frame = np.roll(frame, 5, axis=1)  # simulate right view

        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity_color = stereo_depth_map(left_gray, right_gray)
        annotated = detect_and_annotate(left_frame.copy())
        overlay = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)

        cv2.imshow("DepthMap", np.hstack((disparity_color, overlay)))
        cv2.imshow("Detections", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()