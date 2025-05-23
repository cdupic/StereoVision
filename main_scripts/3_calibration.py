import cv2
import numpy as np
import glob

import os

# Vérifier et créer le répertoire si nécessaire
output_dir = '../calib_result/'
os.makedirs(output_dir, exist_ok=True)


# Paramètres de l'échiquier
rows = 6  # Nombre de coins internes dans les lignes
columns = 9  # Nombre de coins internes dans les colonnes
square_size = 2.5  # Taille d'une case en cm

# Préparation des points 3D de l'échiquier
pattern_size = (columns, rows)
objp = np.zeros((rows * columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size

# Listes pour stocker les points 3D et 2D
objpoints = []  # Points 3D dans le monde réel
imgpoints_left = []  # Points 2D dans les images de la caméra gauche
imgpoints_right = []  # Points 2D dans les images de la caméra droite

# Charger les images gauche et droite
left_images = sorted(glob.glob('../pairs/left_*.png'))
right_images = sorted(glob.glob('../pairs/right_*.png'))

# Détection des coins de l'échiquier
for left_img_path, right_img_path in zip(left_images, right_images):
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size)

    if ret_left and ret_right:
        objpoints.append(objp)
        corners_left = cv2.cornerSubPix(
            gray_left, corners_left, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )
        corners_right = cv2.cornerSubPix(
            gray_right, corners_right, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

# Calibration des caméras
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None
)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None
)

# Calibration stéréo
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
flags = (cv2.CALIB_FIX_INTRINSIC)
ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1], criteria=criteria, flags=flags
)

# Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1], R, T, flags=0
)

# Sauvegarde des résultats
np.savez(os.path.join(output_dir, 'stereo_calib.npz'),
         mtx_left=mtx_left, dist_left=dist_left,
         mtx_right=mtx_right, dist_right=dist_right,
         R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("Calibration terminée et résultats sauvegardés.")


import numpy as np

# Charger le fichier .npz
data = np.load('../calib_result/stereo_calib.npz')

# Accéder aux données
print(data.files)  # Liste des noms des tableaux dans le fichier
print(data['mtx_left'])  # Accéder à un tableau spécifique
