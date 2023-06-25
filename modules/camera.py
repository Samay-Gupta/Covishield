from threading import Thread
import cv2

class Camera:
    def __init__(self):
        self.__cam = None
        self.__frame = None
        self.__running = True
        self.__thread = Thread(target=self.__update_frame)
        self.__thread.setDaemon = False

    def start(self, camera_id=0):
        self.__cam = cv2.VideoCapture(camera_id)
        self.__running = True
        self.__thread.start()

    def get_frame(self):
        return self.__frame

    def __update_frame(self):
        while self.__cam is not None and self.__running:
            ret, frame = self.__cam.read()
            if ret:
                self.__frame = frame

    def stop(self):
        self.__running = False
        self.__thread.join()
        self.__cam.release()