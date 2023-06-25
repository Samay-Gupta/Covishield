from threading import Thread
from .camera import Camera
from .social_distancing_detector import SocialDistancingDetector
from .face_detector import FaceDetector
from .mask_detector import MaskDetector
from .face_classifier import FaceClassifier
from .firebase_link import FirebaseLink
from .static_data import IMAGES_DIR
from time import sleep
import cv2
import os

class SocialTrackApp:
    def __init__(self):
        self.cam = Camera()
        self.cam.start()
        self.running = False
        self.soc_det = SocialDistancingDetector()
        self.face_det = FaceDetector()
        self.mask_det = MaskDetector()
        self.face_clf = FaceClassifier()
        self.storage = FirebaseLink()

    def start(self):
        self.running = True
        self.__run_app()

    def __run_app(self):
        while self.running:
            frame = self.cam.get_frame()
            soc_res = self.soc_det.find_failures(frame)
            result = []
            res = None
            for per_conf, (x, y, dx, dy), fail in soc_res:
                res_data = {
                    "social_distancing": not fail,
                    "person_boundary": [x, y, x+dx, y+dy],
                    "wearing_mask": True,
                    "face_boundary": None,
                    "person_name": "Unknown",
                }
                person = frame[y:y+dy, x:x+dx]
                clr = (0, 255, 0) if not fail else (0, 0, 255)
                frame = cv2.rectangle(frame,(x, y),(x+dx, y+dy),clr,3)
                face_poss = self.face_det.detect(person)
                print(face_poss)
                if len(face_poss) >= 1:
                    face_data = face_poss[0]
                    face_conf, res_data["face_boundary"] = face_data
                    [start_x, start_y, end_x, end_y] = res_data["face_boundary"]
                    face_blob = person[start_y: end_y, start_x: end_x]
                    res_data["wearing_mask"] = self.mask_det.is_wearing(face_blob)
                    res_data["person_name"] = self.face_clf.get_name(face_blob)
                    frame = cv2.rectangle(frame, (x+start_x, y+start_y), (x+end_x, y+end_y), [255, 255, 255], 3)
                    label = "{}: {}".format(res_data["person_name"], 
                        "MASK" if res_data["wearing_mask"] else "NO MASK")
                    cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
                result.append(res_data)
            filename = os.path.join(IMAGES_DIR, "clf_img.png")
            cv2.imwrite(filename, frame)
            self.storage.upload_result(result, filename)
            if frame is not None:
                cv2.imshow("Video", frame)
            if cv2.waitKey(1) == ord('q'):
                self.stop()

    def stop(self):
        self.running = False
        self.cam.stop()
