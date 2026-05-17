import os
import re
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["FLAGS_enable_pir_in_executor"]   = "0"
os.environ["FLAGS_use_mkldnn"]               = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]           = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ===================================================
# CONFIG
# ===================================================

VEHICLE_CONF      = 0.25
PLATE_CONF        = 0.15
TWO_WHEELER_CLASS = 3
DETECT_UPSCALE    = 2.0
PLATE_PAD_PX      = 8

INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN",
    "GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD",
    "MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ",
    "SK","TN","TS","TR","UK","UP","WB",
}

PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")

# ===================================================
# LOAD MODELS
# ===================================================

vehicle_model = YOLO("models/yolo11n.pt")
plate_model   = YOLO("models/best.pt")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    show_log=False,
    use_mp=False,
)

# ===================================================
# HELPERS
# ===================================================

def clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def format_plate(text: str) -> str:
    text  = clean_text(text)
    match = PLATE_RE.match(text)
    if not match:
        return text
    state, district, series, number = match.groups()
    if state not in INDIAN_STATES:
        return text
    return f"{state} {district} {series} {number}"


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def read_plate(crop: np.ndarray) -> str:
    processed = preprocess_plate(crop)

    try:
        results = ocr.ocr(processed, cls=True)
    except Exception as e:
        print(f"  OCR error: {e}")
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

    raw = clean_text("".join(texts))
    if len(raw) < 4:
        return ""

    return format_plate(raw)


def add_padding(box: tuple, source: np.ndarray, pad: int):
    x1, y1, x2, y2 = box
    h, w            = source.shape[:2]
    return source[
        max(0, y1-pad) : min(h, y2+pad),
        max(0, x1-pad) : min(w, x2+pad)
    ], (max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad))


def draw_annotation(img, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, y1-th-14), (x1+tw+8, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ===================================================
# MAIN
# ===================================================

def process_image(image_path: str, output_path: str = "output.jpg"):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot load {image_path}")
        return

    output    = image.copy()
    debug_idx = 0

    vehicle_results = vehicle_model(image, imgsz=1280, conf=VEHICLE_CONF)[0]

    for vbox in vehicle_results.boxes:
        if int(vbox.cls[0]) != TWO_WHEELER_CLASS:
            continue
        if float(vbox.conf[0]) < VEHICLE_CONF:
            continue

        vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
        bike_crop = image[vy1:vy2, vx1:vx2]
        if bike_crop.size == 0:
            continue

        print(f"Bike at ({vx1},{vy1})→({vx2},{vy2})")

        h, w          = bike_crop.shape[:2]
        bike_upscaled = cv2.resize(
            bike_crop,
            (int(w * DETECT_UPSCALE), int(h * DETECT_UPSCALE)),
            interpolation=cv2.INTER_LANCZOS4
        )

        plate_results = plate_model(
            bike_upscaled, imgsz=1280, conf=PLATE_CONF, augment=True
        )[0]

        if len(plate_results.boxes) == 0:
            print("  No plate detected")
            continue

        for pbox in plate_results.boxes:
            pconf = float(pbox.conf[0])

            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            px1 = int(px1 / DETECT_UPSCALE)
            py1 = int(py1 / DETECT_UPSCALE)
            px2 = int(px2 / DETECT_UPSCALE)
            py2 = int(py2 / DETECT_UPSCALE)

            pw, ph = px2 - px1, py2 - py1
            if pw < 20 or ph < 8:
                continue
            if not (1.2 <= pw / max(ph, 1) <= 7.0):
                continue

            gx1, gy1 = vx1 + px1, vy1 + py1
            gx2, gy2 = vx1 + px2, vy1 + py2

            plate_crop, (gx1, gy1, gx2, gy2) = add_padding(
                (gx1, gy1, gx2, gy2), image, PLATE_PAD_PX
            )
            if plate_crop.size == 0:
                continue

            print(f"  Plate crop: {plate_crop.shape[1]}x{plate_crop.shape[0]}px  conf: {pconf:.2f}")

            plate_text = read_plate(plate_crop)

            color = (0, 255, 0) if plate_text else (0, 165, 255)
            label = plate_text  if plate_text else f"Plate {pconf:.2f}"

            print(f"  OCR result: '{plate_text or '-'}'")

            draw_annotation(output, (gx1, gy1, gx2, gy2), label, color)
            debug_idx += 1

    cv2.imwrite(output_path, output)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py test.jpg")
        sys.exit(1)
    process_image(sys.argv[1])
