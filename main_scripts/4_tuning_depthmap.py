import cv2
import numpy as np
import json

loading = False

def stereo_depth_map(left, right, variable_mapping):
    sbm = cv2.StereoBM_create(numDisparities=variable_mapping["NumofDisp"],
                               blockSize=variable_mapping["SWS"])
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(variable_mapping['PreFiltSize'])
    sbm.setPreFilterCap(variable_mapping['PreFiltCap'])
    sbm.setSpeckleRange(variable_mapping['SpeckleRange'])
    sbm.setSpeckleWindowSize(variable_mapping['SpeckleSize'])
    sbm.setMinDisparity(variable_mapping['MinDisp'])
    sbm.setTextureThreshold(variable_mapping['TxtrThrshld'])
    sbm.setUniquenessRatio(variable_mapping['UniqRatio'])

    disparity = sbm.compute(left, right)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype=np.uint8)
    image = cv2.medianBlur(image, 5)
    color_map = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    color_map = cv2.medianBlur(color_map, 5)
    return color_map, disparity_normalized

def activateTrackbars(x):
    global loading
    loading = False

def create_trackbars():
    cv2.createTrackbar("SWS", "Stereo", 15, 255, activateTrackbars)
    cv2.createTrackbar("SpeckleSize", "Stereo", 100, 300, activateTrackbars)
    cv2.createTrackbar("SpeckleRange", "Stereo", 15, 40, activateTrackbars)
    cv2.createTrackbar("UniqRatio", "Stereo", 10, 20, activateTrackbars)
    cv2.createTrackbar("TxtrThrshld", "Stereo", 100, 1000, activateTrackbars)
    cv2.createTrackbar("NumofDisp", "Stereo", 1, 16, activateTrackbars)
    cv2.createTrackbar("MinDisp", "Stereo", 100, 300, activateTrackbars)
    cv2.createTrackbar("PreFiltCap", "Stereo", 30, 63, activateTrackbars)
    cv2.createTrackbar("PreFiltSize", "Stereo", 105, 255, activateTrackbars)

def get_variable_mapping():
    variable_mapping = {}
    for v in ["SWS", "SpeckleSize", "SpeckleRange", "UniqRatio", "TxtrThrshld",
              "NumofDisp", "MinDisp", "PreFiltCap", "PreFiltSize"]:
        val = cv2.getTrackbarPos(v, "Stereo")
        if v in ["SWS", "PreFiltSize"] and (val < 5 or val % 2 == 0):
            val = max(5, val | 1)
        if v == "NumofDisp":
            val = max(1, val) * 16
        if v == "MinDisp":
            val -= 100
        if v in ["UniqRatio", "PreFiltCap"] and val == 0:
            val = 1
        variable_mapping[v] = val
    return variable_mapping

def onMouse(event, x, y, flags, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Distance (pixel value) at ({x},{y}) = {disparity_normalized[y][x]}")

if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # AVFoundation = plus sûr pour macOS

    if not cap.isOpened():
        print("❌ Cannot open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Stereo")
    create_trackbars()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simule une stéréo : gauche = original, droite = image décalée
        left = gray
        right = np.roll(gray, 5, axis=1)  # décale l'image un peu

        variable_mapping = get_variable_mapping()

        disparity_color, disparity_normalized = stereo_depth_map(left, right, variable_mapping)
        cv2.setMouseCallback("Stereo", onMouse, disparity_normalized)

        cv2.imshow("Stereo", disparity_color)
        cv2.imshow("Frame", np.hstack((left, right)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()