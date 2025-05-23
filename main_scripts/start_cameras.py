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


if __name__ == "__main__":

    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()




    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()
        print("flux gauche et droit acquis")

        if left_grabbed and right_grabbed:
            images = np.hstack((left_frame, right_frame))
            cv2.imshow("Camera Images", images)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
        else:
            print("Failed to grab frames from one or both cameras")
            break

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()
