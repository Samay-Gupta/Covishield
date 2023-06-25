from .static_data import FACE_CLF_DIR
import numpy as np
import pickle
import cv2
import os

class FaceClassifier:
    def __init__(self, recognizer_file_path="recognizer.pickle", 
            embeddings_file_path="openface_nn4.small2.v1.t7", 
            encoder_file_path="encoder.pickle"):
        recognizer_file_path = os.path.join(FACE_CLF_DIR, recognizer_file_path)
        embeddings_file_path = os.path.join(FACE_CLF_DIR, embeddings_file_path)
        encoder_file_path = os.path.join(FACE_CLF_DIR, encoder_file_path)
        self.__data_embedder = cv2.dnn.readNetFromTorch(embeddings_file_path)
        with open(encoder_file_path, "rb") as encoder_file_object:
            self.__label_encoder = pickle.load(encoder_file_object)
        with open(recognizer_file_path, "rb") as recognizer_file_object:
            self.__classifier = pickle.load(recognizer_file_object)

    def get_name(self, face_image, confidence=0.3):
        name = "Unknown"
        if min(face_image.shape[:2]) > 20:
            face_image_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), 
                (0, 0, 0), swapRB=True, crop=False)
            self.__data_embedder.setInput(face_image_blob)
            visual_encoder = self.__data_embedder.forward()
            predictions = self.__classifier.predict_proba(visual_encoder)[0]
            res_ind = np.argmax(predictions)
            if predictions[res_ind] > confidence:
                name = self.__label_encoder.classes_[res_ind]
            print(predictions[res_ind])
        return name