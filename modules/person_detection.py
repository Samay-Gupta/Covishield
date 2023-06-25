import numpy as np
import cv2
import os

class PersonDetector:
    def __init__(self):
        self.__labels = []
        self.__classifier = None
        self.__layer_names = None
        self.__threshold = 0.3
        self.__confidence = 0.5

    def load_classifier(self, classifier_folder='storage/yolov3', confidence=0.5,
            threshold=0.3):
        self.__threshold = threshold
        self.__confidence = confidence
        config_file = os.path.join(classifier_folder, 'classifier.cfg')
        labels_file = os.path.join(classifier_folder, 'classifier.names')
        weights_file = os.path.join(classifier_folder, 'classifier.weights')
        print("[INFO] Loading Classifier...")
        try:
            with open(labels_file, 'r') as labels_file_object:
                self.__labels = [label.strip().capitalize() for label in labels_file_object.read().split("\n")]
            self.__classifier = cv2.dnn.readNetFromDarknet(config_file, weights_file)
            layer_names = self.__classifier.getLayerNames()
            unconnected_layers = self.__classifier.getUnconnectedOutLayers()
            self.__layer_names = [layer_names[i[0]-1] for i in unconnected_layers]
            print("[INFO] Classifier Loaded")
        except Exception as LoadError:
            print("[ERROR] Failed to load Classifier")
            print("EXCEPTION RAISED:\n{}\n".format(LoadError))

    def classify(self, frame):
        classification_boxes = []
        classification_class_ids = []
        classification_confidences = []
        required_id = 0
        image_height, image_width = frame.shape[:2]
        box_factor = np.array([image_width, image_height, image_width, image_height])
        image_blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416, 416),
            swapRB=True, crop=False)
        self.__classifier.setInput(image_blob)
        layer_outputs = self.__classifier.forward(self.__layer_names)
        for layer_classifications in layer_outputs:
            for classification in layer_classifications:
                accuracy_scores = classification[5:]
                class_id = np.argmax(accuracy_scores)
                confidence = float(accuracy_scores[class_id])
                if (class_id == required_id and confidence >= self.__confidence):
                    bounding_box = (classification[:4] * box_factor)
                    (center_x, center_y, width, heigth) = bounding_box.astype("int")
                    start_x = int(center_x-(width//2))
                    start_y = int(center_y-(heigth//2))
                    classification_class_ids.append(class_id)
                    classification_confidences.append(confidence)
                    classification_boxes.append([start_x, start_y, int(width), int(heigth)])
        idxs = cv2.dnn.NMSBoxes(classification_boxes, classification_confidences,
            self.__confidence, self.__threshold)
        result = []
        if len(idxs) > 0:
            result = [[confidence, classification_boxes[i][:4]] for i in idxs.flatten()]
        return result
