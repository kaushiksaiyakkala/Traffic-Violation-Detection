import os
import re
import cv2
import warnings
import logging
import numpy as np

from ultralytics import YOLO
from paddleocr import PaddleOCR
from scipy.optimize import linear_sum_assignment


warnings.filterwarnings("ignore")

os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.disable(logging.WARNING)


PERSON_CLASS = 0
BICYCLE_CLASS = 1
MOTORCYCLE_CLASS = 3

CONF_THRESH = 0.30
MAX_RIDERS = 2
OVERLAP_THRESH = 0.15

HELMET_CONF = 0.35
NO_HELMET_MIN_CONF = 0.50

PLATE_CONF = 0.15
DETECT_UPSCALE = 2.0
PLATE_PAD_PX = 8

USE_HUNGARIAN = True
DEBUG_HELMET = False


class TrafficViolationDetector:

    def __init__(self, model_dir="./models"):

        self.det_model = YOLO(
            os.path.join(model_dir, "yolo11l.pt")
        )

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

        self.plate_model = YOLO(
            os.path.join(model_dir, "best.pt")
        )

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            use_mp=False,
            det_model_dir=os.path.join(model_dir, "paddleocr", "det"),
            rec_model_dir=os.path.join(model_dir, "paddleocr", "rec"),
            cls_model_dir=os.path.join(model_dir, "paddleocr", "cls")
        )

    # ========================================================
    # PLATE HELPERS
    # ========================================================

    def clean_text(self, text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def format_plate(self, text):
        return self.clean_text(text)

    def preprocess_plate(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read_plate(self, crop):
        processed = self.preprocess_plate(crop)

        try:
            results = self.ocr.ocr(processed, cls=True)
        except Exception:
            return ""

        if not results or not results[0]:
            return ""

        lines = sorted(
            results[0],
            key=lambda l: (l[0][0][1] + l[0][2][1]) / 2
        )

        texts = []

        for line in lines:
            text, conf = line[1]

            if conf > 0.30:
                texts.append(text)

        if not texts:
            return ""

        raw = self.clean_text("".join(texts))

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
    # BIKE FILTER
    # Consistent with pipeline(2).py
    # ========================================================

    def is_real_motorcycle(self, box, img_w, img_h):
        return True

    # ========================================================
    # RIDER ASSOCIATION
    # ========================================================

    def box_overlap_ratio(self, person_box, bike_box):
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

        person_area = (px2 - px1) * (py2 - py1)

        if person_area == 0:
            return 0.0

        return ((ov_x2 - ov_x1) * (ov_y2 - ov_y1)) / person_area

    def is_eligible(self, person_box, bike_box, img_w=None, img_h=None):
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box

        if img_w and img_h:
            person_area = (px2 - px1) * (py2 - py1)
            image_area = img_w * img_h

            if person_area > 0.12 * image_area:
                return False

        person_w = px2 - px1
        bike_w = bx2 - bx1

        if bike_w > 0 and person_w > bike_w * 2.5:
            return False

        if self.box_overlap_ratio(person_box, bike_box) >= OVERLAP_THRESH:
            return True

        center_x = (px1 + px2) / 2

        if bx1 <= center_x <= bx2 and py2 <= by2:
            return True

        return False

    def hungarian_assign(self, bikes, persons, img_w=None, img_h=None):
        assignment = {i: [] for i in range(len(bikes))}

        if not bikes or not persons:
            return assignment

        n_bikes = len(bikes)
        n_persons = len(persons)

        cost = np.full((n_persons, n_bikes), 1000.0)

        for j, person in enumerate(persons):
            for i, bike in enumerate(bikes):
                overlap = self.box_overlap_ratio(person["box"], bike["box"])

                if self.is_eligible(person["box"], bike["box"], img_w, img_h):
                    cost[j, i] = 1.0 - overlap

        person_indices, bike_indices = linear_sum_assignment(cost)

        assigned = set()

        for pi, bi in zip(person_indices, bike_indices):
            if cost[pi, bi] < 1000.0:
                assignment[bi].append(pi)
                assigned.add(pi)

        for pi, person in enumerate(persons):
            if pi in assigned:
                continue

            best_bi = None
            best_score = -1

            for bi, bike in enumerate(bikes):
                overlap = self.box_overlap_ratio(person["box"], bike["box"])

                if (
                    self.is_eligible(person["box"], bike["box"], img_w, img_h)
                    and overlap > best_score
                ):
                    best_score = overlap
                    best_bi = bi

            if best_bi is not None:
                assignment[best_bi].append(pi)
                assigned.add(pi)

        return assignment

    def greedy_assign(self, bikes, persons, img_w=None, img_h=None):
        assignment = {i: [] for i in range(len(bikes))}
        assigned = set()

        pairs = []

        for pi, person in enumerate(persons):
            for bi, bike in enumerate(bikes):
                if not self.is_eligible(person["box"], bike["box"], img_w, img_h):
                    continue

                overlap = self.box_overlap_ratio(person["box"], bike["box"])
                pairs.append((overlap, pi, bi))

        pairs.sort(reverse=True)

        for overlap, pi, bi in pairs:
            if pi in assigned:
                if overlap >= 0.30:
                    assignment[bi].append(pi)
                continue

            assignment[bi].append(pi)
            assigned.add(pi)

        return assignment

    # ========================================================
    # HELMET DETECTION
    # ========================================================

    def run_helmet_on_full_image(self, helmet_model, image, model_name=""):
        results = helmet_model(
            image,
            conf=0.20,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            name = helmet_model.names.get(cls, str(cls))

            detections.append({
                "box": [x1, y1, x2, y2],
                "cls": cls,
                "conf": conf,
                "name": name,
                "model": model_name
            })

        return detections

    def run_helmet_ensemble(self, image, img_w, img_h):
        all_detections = []

        scale = 1.0
        h_img = image

        if img_w > 1280:
            scale = 1280 / img_w
            h_img = cv2.resize(image, (1280, int(img_h * scale)))

        for hcfg in self.helmet_models:
            dets = self.run_helmet_on_full_image(
                hcfg["model"],
                h_img,
                hcfg["name"]
            )

            for d in dets:
                x1, y1, x2, y2 = d["box"]

                d["box"] = [
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale)
                ]

                d["std_cls"] = (
                    "with_helmet"
                    if d["cls"] == hcfg["with_helmet"]
                    else "no_helmet"
                )

            all_detections.extend(dets)

        return all_detections

    def match_helmet_to_rider(self, helmet_detections, person_box, bike_box, rider_idx=0):
        px1, py1, px2, py2 = person_box

        ph = py2 - py1
        pw = px2 - px1

        head_cx = (px1 + px2) / 2
        head_cy = py1 + int(ph * 0.15)

        search_x1 = max(0, px1 - int(pw * 0.50))
        search_y1 = max(0, py1 - int(ph * 1.00))
        search_x2 = px2 + int(pw * 0.50)
        search_y2 = py2

        best_with_score = 0.0
        best_no_score = 0.0
        matched = []

        for det in helmet_detections:
            dx1, dy1, dx2, dy2 = det["box"]

            cx = (dx1 + dx2) / 2
            cy = (dy1 + dy2) / 2

            if not (search_x1 <= cx <= search_x2 and search_y1 <= cy <= search_y2):
                continue

            dist = ((cx - head_cx) ** 2 + (cy - head_cy) ** 2) ** 0.5
            score = det["conf"] / (1.0 + dist / max(pw, 1))

            matched.append({**det, "score": score})

            if det["std_cls"] == "with_helmet" and score > best_with_score:
                best_with_score = score

            if det["std_cls"] == "no_helmet" and score > best_no_score:
                best_no_score = score

        if DEBUG_HELMET:
            if matched:
                det_str = ", ".join(
                    f"{d['name']}={d['conf']:.2f}(s={d['score']:.2f})"
                    for d in matched
                )
                print(f"Rider [{rider_idx}] detections: {det_str}")
            else:
                print(
                    f"Rider [{rider_idx}] NONE | "
                    f"person={person_box} "
                    f"search=[{search_x1},{search_y1},{search_x2},{search_y2}]"
                )

        if best_no_score >= NO_HELMET_MIN_CONF and best_no_score > best_with_score:
            return "no_helmet"

        if best_with_score >= HELMET_CONF:
            return "helmet"

        if matched:
            if best_with_score >= best_no_score:
                return "helmet"
            return "no_helmet"

        if bike_box is not None:
            bx1, by1, bx2, by2 = bike_box

            fb_with = 0.0
            fb_no = 0.0
            fb_matched = []

            for det in helmet_detections:
                dx1, dy1, dx2, dy2 = det["box"]

                cx = (dx1 + dx2) / 2
                cy = (dy1 + dy2) / 2

                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    fb_matched.append(det)

                    if det["std_cls"] == "with_helmet" and det["conf"] > fb_with:
                        fb_with = det["conf"]

                    if det["std_cls"] == "no_helmet" and det["conf"] > fb_no:
                        fb_no = det["conf"]

            if DEBUG_HELMET and fb_matched:
                det_str = ", ".join(
                    f"{d['name']}={d['conf']:.2f}"
                    for d in fb_matched
                )
                print(f"Rider [{rider_idx}] bike-box fallback: {det_str}")

            if fb_no >= NO_HELMET_MIN_CONF and fb_no > fb_with:
                return "no_helmet"

            if fb_with >= HELMET_CONF:
                return "helmet"

            if fb_matched:
                if fb_with >= fb_no:
                    return "helmet"
                return "no_helmet"

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

            TARGET_W = 1280
            effective_conf = CONF_THRESH
            det_image = image

            if img_w < 400 or img_h < 300:
                effective_conf = 0.20
                scale = max(640 / img_w, 640 / img_h)

                det_image = cv2.resize(
                    image,
                    (
                        int(img_w * scale),
                        int(img_h * scale)
                    )
                )

            elif img_w > TARGET_W:
                scale = TARGET_W / img_w

                det_image = cv2.resize(
                    image,
                    (
                        TARGET_W,
                        int(img_h * scale)
                    )
                )

            results = self.det_model(
                det_image,
                conf=effective_conf,
                verbose=False
            )[0]

            det_scale_x = img_w / det_image.shape[1]
            det_scale_y = img_h / det_image.shape[0]

            bikes = []
            persons = []

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = int(x1 * det_scale_x)
                y1 = int(y1 * det_scale_y)
                x2 = int(x2 * det_scale_x)
                y2 = int(y2 * det_scale_y)

                if cls in (MOTORCYCLE_CLASS, BICYCLE_CLASS):
                    if self.is_real_motorcycle([x1, y1, x2, y2], img_w, img_h):
                        bikes.append({
                            "box": [x1, y1, x2, y2],
                            "conf": conf
                        })

                elif cls == PERSON_CLASS:
                    persons.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf
                    })

            helmet_detections = []

            if self.helmet_models:
                helmet_detections = self.run_helmet_ensemble(
                    image,
                    img_w,
                    img_h
                )

            if USE_HUNGARIAN:
                assignment = self.hungarian_assign(
                    bikes,
                    persons,
                    img_w,
                    img_h
                )
            else:
                assignment = self.greedy_assign(
                    bikes,
                    persons,
                    img_w,
                    img_h
                )

            violations = []

            for i, bike in enumerate(bikes):
                bx1, by1, bx2, by2 = bike["box"]

                person_indices = assignment[i]
                rider_boxes = [
                    persons[j]["box"]
                    for j in person_indices
                ]

                rider_count = len(rider_boxes)

                helmet_violations = 0

                for rider_idx, rider_box in enumerate(rider_boxes):
                    if self.helmet_models:
                        status = self.match_helmet_to_rider(
                            helmet_detections,
                            rider_box,
                            [bx1, by1, bx2, by2],
                            rider_idx
                        )
                    else:
                        status = "unknown"

                    if status == "no_helmet":
                        helmet_violations += 1

                is_triple = rider_count > MAX_RIDERS
                is_no_helmet = helmet_violations > 0

                # Only output violations
                if not is_triple and not is_no_helmet:
                    continue

                bike_crop = image[by1:by2, bx1:bx2]
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
                        conf=PLATE_CONF,
                        verbose=False
                    )[0]

                    best_conf = 0
                    best_crop = None

                    for pbox in plate_results.boxes:
                        conf = float(pbox.conf[0])

                        px1, py1, px2, py2 = map(
                            int,
                            pbox.xyxy[0]
                        )

                        px1 = int(px1 / DETECT_UPSCALE)
                        py1 = int(py1 / DETECT_UPSCALE)
                        px2 = int(px2 / DETECT_UPSCALE)
                        py2 = int(py2 / DETECT_UPSCALE)

                        pw = px2 - px1
                        ph = py2 - py1

                        if pw < 20 or ph < 8:
                            continue

                        if not (1.2 <= pw / max(ph, 1) <= 7.0):
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
                        license_plate = self.read_plate(best_crop)

                except Exception:
                    pass

                violation_types = []

                if is_triple:
                    violation_types.append("triple_riding")

                if is_no_helmet:
                    violation_types.append("no_helmet")

                violations.append({
                    "violation_types": violation_types,
                    "num_riders": int(rider_count),
                    "helmet_violations": int(helmet_violations),
                    "license_plate": str(license_plate)
                })

            return {
                "violations": violations
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "violations": []
            }


if __name__ == "__main__":
    image_path = "test.jpg"

    detector = TrafficViolationDetector(model_dir="./models")
    result = detector.predict(image_path)

    print(result)