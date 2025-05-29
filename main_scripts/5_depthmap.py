import cv2
import numpy as np
import os
import json
from start_cameras import Start_Cameras

# Paramètres par défaut
SWS = 15
PFS = 5
PFC = 29
MDS = 0
NOD = 96
TTH = 100
UR = 10
SR = 14
SPWS = 100
sbm = None  # StereoBM sera initialisé après chargement

def load_map_settings(file):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, sbm
    if not os.path.isfile(file):
        print(f"⚠️ Settings file not found: {file}")
        print("🛠️ Using default parameters.")
    else:
        print('📥 Loading parameters from file...')
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
        print('✅ Parameters loaded.')

    # Initialisation du StereoBM
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

def stereo_depth_map(left_gray, right_gray):
    left = cv2.GaussianBlur(left_gray, (5, 5), 0)
    right = cv2.GaussianBlur(right_gray, (5, 5), 0)
    disparity = sbm.compute(left, right)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_filtered = cv2.medianBlur(np.uint8(disparity_normalized), 5)
    disparity_color = cv2.applyColorMap(disparity_filtered, cv2.COLORMAP_TURBO)
    return disparity_color, disparity_normalized

def onMouse(event, x, y, flags, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        value = disparity_normalized[y][x]
        if value > 0:
            print(f"📏 Disparity at ({x}, {y}): {value:.2f}")
        else:
            print(f"📏 Invalid disparity at ({x}, {y})")

if __name__ == "__main__":
    # Utiliser 1 seule caméra
    cam = Start_Cameras(0).start()

    # Charger les paramètres
    load_map_settings("../3dmap_set.txt")

    cv2.namedWindow("DepthMap")

    while True:
        grabbed, frame = cam.read()
        if not grabbed:
            continue

        left_frame = frame.copy()
        right_frame = np.roll(frame, 5, axis=1)  # simulation d'une seconde caméra décalée

        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity_color, disparity_normalized = stereo_depth_map(left_gray, right_gray)

        if disparity_color.shape[:2] != left_frame.shape[:2]:
            disparity_color = cv2.resize(disparity_color, (left_frame.shape[1], left_frame.shape[0]))

        output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)

        cv2.setMouseCallback("DepthMap", onMouse, disparity_normalized)
        cv2.imshow("DepthMap", np.hstack((disparity_color, output)))
        cv2.imshow("Frames", np.hstack((left_frame, right_frame)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()