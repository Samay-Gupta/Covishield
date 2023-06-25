from .person_detection import PersonDetector
from .static_methods import get_center, get_distance, calculate_social_dist
from .static_data import CLF_DIR, REQ_DIST

class SocialDistancingDetector:
    def __init__(self):
        self.clf = PersonDetector()
        self.clf.load_classifier(CLF_DIR)

    def find_failures(self, frame):
        clf_list = self.clf.classify(frame)
        centroids = [get_center(rect) for _, rect in clf_list]
        person_count = len(clf_list)
        failures = [False,] * person_count
        if person_count >= 2:
            for i in range(person_count):
                for j in range(person_count):
                    if i==j:
                        continue
                    dist = calculate_social_dist(clf_list[i][1], clf_list[j][1])
                    if dist < REQ_DIST:
                        failures[i] = True
                        failures[j] = True
        return [clf_list[i] + [failures[i]] for i in range(person_count)]
        