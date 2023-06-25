from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from .static_data import MASK_DET_DIR
import numpy as np
import cv2
import os

class MaskDetector:
    def __init__(self):
        self.model = load_model(os.path.join(MASK_DET_DIR, "mask_detector.model"))

    def is_wearing(self, face_blob):
        face_blob = cv2.cvtColor(face_blob, cv2.COLOR_BGR2RGB)
        face_blob = cv2.resize(face_blob, (224, 224))
        face_blob = img_to_array(face_blob)
        face_blob = preprocess_input(face_blob)
        face_blob = np.expand_dims(face_blob, axis=0)
        (mask_prob, without_mask_prob) = self.model.predict(face_blob)[0]
        return mask_prob > without_mask_prob
