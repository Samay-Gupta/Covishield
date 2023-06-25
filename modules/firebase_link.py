from .static_data import FIREBASE_DIR
from .static_methods import get_location, get_date_and_time
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage


class FirebaseLink:
    def __init__(self):
        firebase_file = os.path.join(FIREBASE_DIR, "serviceAccountKey.json")
        cred = credentials.Certificate(firebase_file)
        firebase_admin.initialize_app(cred, {"storageBucket": "covishield-8c42f.appspot.com"})
        self.db = firestore.client()
        self.ds = storage.bucket()
        self.ind = 0

    def upload_result(self, result, image_file):
        file_path = self.upload_image(image_file)
        lat, lng = get_location()
        for res in result:
            violations = []
            if not res["social_distancing"]:
                violations.append("Not Social Distancing")
            if not res["wearing_mask"]:
                violations.append("Not Wearing Mask")
            dt, tm = get_date_and_time()
            if len(violations) > 0:
                self.upload_dict({
                    "violater": res["person_name"],
                    "violation": ", ".join(violations),
                    "imagepath": file_path,
                    "lng": lng,
                    "lat": lat,
                    "time": tm,
                    "date": dt
                })

    def upload_dict(self, data):
        self.db.collection('violations').add(data)

    def upload_image(self, image_file):
        res_filename = f"Violation-{self.ind}.png"
        blob = self.ds.blob(res_filename)
        blob.upload_from_filename(image_file)
        self.ind += 1
        return res_filename