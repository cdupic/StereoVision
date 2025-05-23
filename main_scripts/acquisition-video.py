import os

import cv2
import numpy as np
import threading


class Start_Cameras:

	def __init__(self, camera_index):
		self.video_capture = None
		self.frame = None
		self.grabbed = False
		self.read_thread = None
		self.read_lock = threading.Lock()
		self.running = False
		self.camera_index = camera_index
		self.open()

	def open(self):
		try:
			self.video_capture = cv2.VideoCapture(self.camera_index)
			if not self.video_capture.isOpened():
				print(f"Failed to open camera {self.camera_index}")
				raise RuntimeError(f"Unable to open camera {self.camera_index}")
			self.grabbed, self.frame = self.video_capture.read()
			print(f"Camera {self.camera_index} is opened")
		except RuntimeError as e:
			self.video_capture = None
			print(e)

	def start(self):
		if self.running:
			print('Video capturing is already running')
			return None
		if self.video_capture is not None:
			self.running = True
			self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
			self.read_thread.start()
		return self

	def stop(self):
		self.running = False
		if self.read_thread is not None:
			self.read_thread.join()

	def updateCamera(self):
		while self.running:
			try:
				grabbed, frame = self.video_capture.read()
				with self.read_lock:
					self.grabbed = grabbed
					self.frame = frame
			except RuntimeError:
				print(f"Could not read image from camera {self.camera_index}")

	def read(self):
		with self.read_lock:
			frame = self.frame.copy() if self.frame is not None else None
			grabbed = self.grabbed
		return grabbed, frame

	def release(self):
		if self.video_capture is not None:
			self.video_capture.release()
			self.video_capture = None
		if self.read_thread is not None:
			self.read_thread.join()


	def get(self, prop_id):
		if self.video_capture is not None:
			return self.video_capture.get(prop_id)
		else:
			raise RuntimeError("Camera is not initialized")


if __name__ == "__main__":

	RECORD_OUTPUT_VIDEO = True
	NUMERO_TEST = 2

	output_dir = f'output/video/test{NUMERO_TEST}'
	os.makedirs(output_dir, exist_ok=True)

	output_video_left_path = os.path.join(output_dir, 'left.avi')
	output_video_right_path = os.path.join(output_dir, 'right.avi')
	#output_video_merged_path = os.path.join(output_dir, 'merged.avi')


	fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec vidéo

	left_camera = Start_Cameras(0).start()
	right_camera = Start_Cameras(1).start()


	fps = int(left_camera.get(cv2.CAP_PROP_FPS))  # Fréquence d'images de la vidéo d'entrée
	frame_size = (int(left_camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(left_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

	out_left = cv2.VideoWriter(output_video_left_path, fourcc, fps, frame_size)
	out_right = cv2.VideoWriter(output_video_right_path, fourcc, fps, frame_size)
	#out_merged = cv2.VideoWriter(output_video_merged_path, fourcc, fps, frame_size*2)




	while True:
		left_grabbed, left_frame = left_camera.read()
		right_grabbed, right_frame = right_camera.read()

		if left_grabbed and right_grabbed:
			images = np.hstack((left_frame, right_frame))
			#cv2.imshow("Camera Images", images)
			k = cv2.waitKey(1) & 0xFF


			if RECORD_OUTPUT_VIDEO:
				out_left.write(left_frame)
				out_right.write(right_frame)
				#out_merged.write(images)


			if k == ord('q'):
				break
		else:
			print("Failed to grab frames from one or both cameras")
			break

	left_camera.stop()
	left_camera.release()
	right_camera.stop()
	right_camera.release()

	if RECORD_OUTPUT_VIDEO:
		out_left.release()
		out_right.release()
		#out_merged.release()
	cv2.destroyAllWindows()
