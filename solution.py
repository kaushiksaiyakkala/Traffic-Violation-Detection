import os
import re
import cv2
import warnings
import logging
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR

# ============================================================
# ENV FIXES
# ============================================================

warnings.filterwarnings("ignore")

os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ============================================================
# CONFIG
# ============================================================

PERSON_CLASS = 0
BICYCLE_CLASS = 1
MOTORCYCLE_CLASS = 3

CONF_THRESH = 0.30

MAX_RIDERS = 2

OVERLAP_THRESH = 0.15

MAX_BIKE_WIDTH_FRACTION = 0.22
MAX_ASPECT_RATIO = 1.4

HELMET_CONF = 0.35
NO_HELMET_MIN_CONF = 0.50

PLATE_CONF = 0.15
DETECT_UPSCALE = 2.0
PLATE_PAD_PX = 8

INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN",
    "GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD",
    "MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ",
    "SK","TN","TS","TR","UK","UP","WB",
}

PLATE_RE = re.compile(
    r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$"
)

# ============================================================
# MAIN CLASS
# ============================================================

class TrafficViolationDetector:

    def __init__(self, model_dir="./models"):

        # ----------------------------------------------------
        # Detection model
        # ----------------------------------------------------

        self.det_model = YOLO(
            os.path.join(model_dir, "yolo11n.pt")
        )

        # ----------------------------------------------------
        # Helmet models
        # ----------------------------------------------------

        self.helmet_models = []

        helmet_configs = [
            {
                "path": os.path.join(model_dir, "helmet_github.pt"),
                "with_helmet": 0,
                "no_helmet": 1,
                "name": "github"
            },
            {
                "path": os.path.join(model_dir, "helmet_iamtsr.pt"),
                "with_helmet": 0,
                "no_helmet": 1,
                "name": "iam-tsr"
            }
        ]

        for cfg in helmet_configs:

            if os.path.exists(cfg["path"]):

                model = YOLO(cfg["path"])

                self.helmet_models.append({
                    **cfg,
                    "model": model
                })

        # ----------------------------------------------------
        # Plate detector
        # ----------------------------------------------------

        self.plate_model = YOLO(
            os.path.join(model_dir, "best.pt")
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            use_mp=False,

            det_model_dir=os.path.join(
                model_dir,
                "paddleocr",
                "det"
            ),

            rec_model_dir=os.path.join(
                model_dir,
                "paddleocr",
                "rec"
            ),

            cls_model_dir=os.path.join(
                model_dir,
                "paddleocr",
                "cls"
            )
        )

    # ========================================================
    # HELPERS
    # ========================================================

    def clean_text(self, text):

        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def format_plate(self, text):

        text = self.clean_text(text)

        match = PLATE_RE.match(text)

        if not match:
            return text

        state, district, series, number = match.groups()

        if state not in INDIAN_STATES:
            return text

        return f"{state}{district}{series}{number}"

    def preprocess_plate(self, crop):

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.bilateralFilter(
            gray,
            9,
            75,
            75
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

    def read_plate(self, crop):

        processed = self.preprocess_plate(crop)

        try:

            results = self.ocr.ocr(
                processed,
                cls=True
            )

        except:
            return ""

        if not results or not results[0]:
            return ""

        lines = sorted(
            results[0],
            key=lambda l: (
                l[0][0][1] + l[0][2][1]
            ) / 2
        )

        texts = []

        for line in lines:

            text, conf = line[1]

            if conf > 0.30:
                texts.append(text)

        if not texts:
            return ""

        raw = self.clean_text(
            "".join(texts)
        )

        if len(raw) < 4:
            return ""

        return self.format_plate(raw)

    def add_padding(self, box, source, pad):

        x1, y1, x2, y2 = box

        h, w = source.shape[:2]

        return source[
            max(0, y1 - pad):min(h, y2 + pad),
            max(0, x1 - pad):min(w, x2 + pad)
        ]

    # ========================================================
    # MOTORCYCLE FILTER
    # ========================================================

    def is_real_motorcycle(self, box, img_w, img_h):

        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1

        if bh == 0:
            return False

        if (bw / bh) > MAX_ASPECT_RATIO:
            return False

        if (bw / img_w) > MAX_BIKE_WIDTH_FRACTION:
            return False

        return True

    # ========================================================
    # RIDER ASSOCIATION
    # ========================================================

    def box_overlap_ratio(
        self,
        person_box,
        bike_box
    ):

        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box

        ov_y1 = max(py1, by1)
        ov_y2 = min(py2, by2)

        if ov_y2 <= ov_y1:
            return 0.0

        ov_x1 = max(px1, bx1)
        ov_x2 = min(px2, bx2)

        if ov_x2 <= ov_x1:
            return 0.0

        person_area = (
            (px2 - px1)
            * (py2 - py1)
        )

        if person_area == 0:
            return 0.0

        return (
            ((ov_x2 - ov_x1)
            * (ov_y2 - ov_y1))
            / person_area
        )

    def is_eligible(
        self,
        person_box,
        bike_box
    ):

        overlap = self.box_overlap_ratio(
            person_box,
            bike_box
        )

        return overlap >= OVERLAP_THRESH

    # ========================================================
    # HELMET ENSEMBLE
    # ========================================================

    def run_helmet_ensemble(self, image):

        detections = []

        for cfg in self.helmet_models:

            results = cfg["model"](
                image,
                conf=0.20,
                verbose=False
            )[0]

            for box in results.boxes:

                cls = int(box.cls[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": float(box.conf[0]),
                    "type": (
                        "helmet"
                        if cls == cfg["with_helmet"]
                        else "no_helmet"
                    )
                })

        return detections

    def match_helmet_to_rider(
        self,
        helmet_detections,
        person_box
    ):

        px1, py1, px2, py2 = person_box

        best_helmet = 0
        best_no_helmet = 0

        for det in helmet_detections:

            dx1, dy1, dx2, dy2 = det["box"]

            cx = (dx1 + dx2) / 2
            cy = (dy1 + dy2) / 2

            if not (px1 <= cx <= px2):
                continue

            if not (py1 <= cy <= py2):
                continue

            if det["type"] == "helmet":

                best_helmet = max(
                    best_helmet,
                    det["conf"]
                )

            else:

                best_no_helmet = max(
                    best_no_helmet,
                    det["conf"]
                )

        if (
            best_no_helmet >= NO_HELMET_MIN_CONF
            and best_no_helmet > best_helmet
        ):
            return "no_helmet"

        if best_helmet >= HELMET_CONF:
            return "helmet"

        return "unknown"

    # ========================================================
    # MAIN PREDICT
    # ========================================================

    def predict(self, image_path):

        try:

            image = cv2.imread(image_path)

            if image is None:

                return {
                    "violations": []
                }

            img_h, img_w = image.shape[:2]

            # ------------------------------------------------
            # Small image handling
            # ------------------------------------------------

            effective_conf = CONF_THRESH
            det_image = image

            if img_w < 500 or img_h < 400:

                effective_conf = 0.20

                scale = max(
                    640 / img_w,
                    640 / img_h
                )

                det_image = cv2.resize(
                    image,
                    (
                        int(img_w * scale),
                        int(img_h * scale)
                    )
                )

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            results = self.det_model(
                det_image,
                conf=effective_conf
            )[0]

            det_scale_x = (
                img_w / det_image.shape[1]
            )

            det_scale_y = (
                img_h / det_image.shape[0]
            )

            bikes = []
            persons = []

            # ------------------------------------------------
            # Extract bikes and persons
            # ------------------------------------------------

            for box in results.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = int(x1 * det_scale_x)
                y1 = int(y1 * det_scale_y)
                x2 = int(x2 * det_scale_x)
                y2 = int(y2 * det_scale_y)

                if cls in [
                    MOTORCYCLE_CLASS,
                    BICYCLE_CLASS
                ]:

                    if self.is_real_motorcycle(
                        [x1, y1, x2, y2],
                        img_w,
                        img_h
                    ):

                        bikes.append({
                            "box": [x1, y1, x2, y2],
                            "conf": conf
                        })

                elif cls == PERSON_CLASS:

                    persons.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf
                    })

            # ------------------------------------------------
            # Helmet detections
            # ------------------------------------------------

            helmet_detections = (
                self.run_helmet_ensemble(image)
            )

            violations = []

            # ------------------------------------------------
            # Process each bike
            # ------------------------------------------------

            for bike in bikes:

                bx1, by1, bx2, by2 = bike["box"]

                associated_persons = []

                for person in persons:

                    if self.is_eligible(
                        person["box"],
                        bike["box"]
                    ):

                        associated_persons.append(
                            person
                        )

                rider_count = len(
                    associated_persons
                )

                helmet_violations = 0

                # --------------------------------------------
                # Helmet checking
                # --------------------------------------------

                for person in associated_persons:

                    status = (
                        self.match_helmet_to_rider(
                            helmet_detections,
                            person["box"]
                        )
                    )

                    if status == "no_helmet":
                        helmet_violations += 1

                # --------------------------------------------
                # Skip non-violating vehicles
                # --------------------------------------------

                if (
                    rider_count <= MAX_RIDERS
                    and helmet_violations == 0
                ):
                    continue

                # --------------------------------------------
                # Plate OCR
                # --------------------------------------------

                bike_crop = image[
                    by1:by2,
                    bx1:bx2
                ]

                license_plate = ""

                try:

                    h, w = bike_crop.shape[:2]

                    if h == 0 or w == 0:
                        continue

                    bike_upscaled = cv2.resize(
                        bike_crop,
                        (
                            int(w * DETECT_UPSCALE),
                            int(h * DETECT_UPSCALE)
                        ),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                    plate_results = self.plate_model(
                        bike_upscaled,
                        imgsz=1280,
                        conf=PLATE_CONF
                    )[0]

                    best_conf = 0
                    best_crop = None

                    for pbox in plate_results.boxes:

                        conf = float(
                            pbox.conf[0]
                        )

                        px1, py1, px2, py2 = map(
                            int,
                            pbox.xyxy[0]
                        )

                        px1 = int(
                            px1 / DETECT_UPSCALE
                        )

                        py1 = int(
                            py1 / DETECT_UPSCALE
                        )

                        px2 = int(
                            px2 / DETECT_UPSCALE
                        )

                        py2 = int(
                            py2 / DETECT_UPSCALE
                        )

                        pw = px2 - px1
                        ph = py2 - py1

                        # Geometry filtering

                        if pw < 20 or ph < 8:
                            continue

                        if not (
                            1.2 <=
                            pw / max(ph, 1)
                            <= 7.0
                        ):
                            continue

                        if conf > best_conf:

                            best_conf = conf

                            best_crop = self.add_padding(
                                (
                                    px1,
                                    py1,
                                    px2,
                                    py2
                                ),
                                bike_crop,
                                PLATE_PAD_PX
                            )

                    if best_crop is not None:

                        license_plate = (
                            self.read_plate(
                                best_crop
                            )
                        )

                except:
                    pass

                violations.append({

                    "num_riders": int(
                        rider_count
                    ),

                    "helmet_violations": int(
                        helmet_violations
                    ),

                    "license_plate": str(
                        license_plate
                    )
                })

            return {
                "violations": violations
            }

        except Exception:

            return {
                "violations": []
            }