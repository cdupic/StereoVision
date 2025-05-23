import cv2
import numpy as np

# Charger les vidéos gauche et droite
left_video_path = "output/video/test2/left.avi"
right_video_path = "output/video/test2/right.avi"

left_cap = cv2.VideoCapture(left_video_path)
right_cap = cv2.VideoCapture(right_video_path)

# Vérifier si les vidéos sont ouvertes
if not left_cap.isOpened() or not right_cap.isOpened():
	print("Erreur lors de l'ouverture des vidéos.")
	exit()

# Initialiser le StereoBM pour la création de la depth map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
	ret_left, left_frame = left_cap.read()
	ret_right, right_frame = right_cap.read()

	# Vérifier si les frames ont été correctement lues
	if not ret_left or not ret_right:
		print("Fin des vidéos ou erreur de lecture.")
		break

	# Convertir les frames en niveaux de gris
	left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
	right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

	# Calculer la depth map
	disparity = stereo.compute(left_gray, right_gray)
	disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
	depth_map = np.uint8(disparity_normalized)

	# Afficher la depth map
	cv2.imshow("Depth Map", depth_map)

	# Quitter avec la touche 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Libérer les ressources
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
