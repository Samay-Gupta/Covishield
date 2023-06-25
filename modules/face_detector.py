from .static_data import FACE_DET_DIR
import numpy as np
import cv2
import os

class FaceDetector:
    def __init__(self, caffe_file_path="res_ssd_300x300.caffemodel", 
            prototxt_file_path="deploy.prototxt"):
        caffe_file_path = os.path.join(FACE_DET_DIR, caffe_file_path)
        prototxt_file_path = os.path.join(FACE_DET_DIR, prototxt_file_path)
        self.__detector = cv2.dnn.readNetFromCaffe(
            prototxt_file_path,
            caffe_file_path
        )

    def detect(self, image_object, min_confidence=0.5):
        if min(image_object.shape[:2]) <= 20:
            return []
        image_height, image_width = image_object.shape[:2]
        box_factor = np.array([image_width, image_height, image_width, image_height])
        detections = []
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image_object, (300, 300)), 1.0, 
            (300, 300), (104.0, 177.0, 23.0))
        self.__detector.setInput(image_blob)
        face_detections = self.__detector.forward()
        for i in range(face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence >= min_confidence:
                bounding_box = (face_detections[0, 0, i, 3:7] * box_factor).astype("int")
                (start_x, start_y, end_x, end_y) = bounding_box
                detections.append([confidence, [start_x, start_y, end_x, end_y]])
        return detections