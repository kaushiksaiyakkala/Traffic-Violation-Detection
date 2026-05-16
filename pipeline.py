from ultralytics import YOLO
import cv2
import os

# =============================================================================
# CONFIG
# =============================================================================

IMAGE_PATH = r"C:\Users\kaush\Downloads\Traffic-Violation-Detection\Images to test on\WhatsApp Image 2026-05-14 at 8.41.08 PM.jpeg"

# Set to "github", "huggingface", or "iam-tsr"
HELMET_MODEL_CHOICE = "github"

# Set to True to print every helmet detection result for debugging
DEBUG_HELMET = True

# =============================================================================
# Fixed config
# =============================================================================

DETECTION_MODEL_PATH = "models/yolo11n.pt"
CONF_THRESH  = 0.30
MAX_RIDERS   = 2

PERSON_CLASS     = 0
BICYCLE_CLASS    = 1
MOTORCYCLE_CLASS = 3

OVERLAP_THRESH          = 0.25
MAX_BIKE_WIDTH_FRACTION = 0.25
MAX_ASPECT_RATIO        = 1.4

# Confidence thresholds for helmet detection
# Run inference at 0.20 to catch all detections, then filter by class:
# "With Helmet" needs 0.30 to confirm
# "No Helmet" needs 0.50 AND must beat With Helmet score (avoid false violations)
HELMET_CONF         = 0.30   # min conf to confirm "Helmet OK"
NO_HELMET_MIN_CONF  = 0.50   # min conf to flag a violation

# Class IDs per model
# GitHub (Juliowiwiwiwi): 0=With Helmet, 1=Without Helmet
# HuggingFace (aneesarom): 0=Rider, 1=With Helmet, 2=Without Helmet, 3=Number Plate
HELMET_CLASSES = {
    "github": {
        "path":        "models/helmet_github.pt",
        "with_helmet": 0,
        "no_helmet":   1,
    },
    "huggingface": {
        "path":        "models/helmet_hf.pt",
        "with_helmet": 1,
        "no_helmet":   2,
    },
    "iam-tsr": {
        "path":        "models/helmet_iamtsr.pt",
        "with_helmet": 0,
        "no_helmet":   1,
    },
}

# =============================================================================
# Motorcycle filter
# =============================================================================

def is_real_motorcycle(box, img_w, img_h):
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

# =============================================================================
# Rider association
# =============================================================================

def box_overlap_ratio(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    ov_y1 = max(py1, by1);  ov_y2 = min(py2, by2)
    if ov_y2 <= ov_y1: return 0.0
    ov_x1 = max(px1, bx1);  ov_x2 = min(px2, bx2)
    if ov_x2 <= ov_x1: return 0.0
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0: return 0.0
    return ((ov_x2 - ov_x1) * (ov_y2 - ov_y1)) / person_area

def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    if box_overlap_ratio(person_box, bike_box) >= OVERLAP_THRESH:
        return True
    center_x = (px1 + px2) / 2
    bike_h   = by2 - by1
    if bx1 <= center_x <= bx2 and by1 - bike_h * 0.5 <= py2 <= by2 + 20:
        return True
    return False

# =============================================================================
# Helmet detection — run ONCE on full image, match detections to riders
# =============================================================================

def run_helmet_on_full_image(helmet_model, image):
    """
    Run helmet model on the full image once.
    Returns list of dicts: {box, cls, conf, name}
    Much better than cropping — model sees full context at proper resolution.
    """
    results = helmet_model(image, conf=0.20, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = helmet_model.names.get(cls, str(cls))
        detections.append({"box": [x1, y1, x2, y2], "cls": cls, "conf": conf, "name": name})
    return detections


def match_helmet_to_rider(helmet_detections, person_box, with_cls, no_cls, rider_idx):
    """
    Find helmet detections whose center falls inside the person box
    (expanded upward 60% to cover head region above the person box).
    Returns: "helmet", "no_helmet", or "unknown"
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1

    # Search zone: expand upward 100% and sideways 20%
    # Helmet box center must fall in this zone to be matched to this rider
    search_x1 = max(0, px1 - int(pw * 0.20))
    search_y1 = max(0, py1 - int(ph * 1.00))
    search_x2 = px2 + int(pw * 0.20)
    search_y2 = py2

    best_with_conf = 0.0
    best_no_conf   = 0.0
    matched = []

    for det in helmet_detections:
        dx1, dy1, dx2, dy2 = det["box"]
        # Use center of helmet detection box
        cx = (dx1 + dx2) / 2
        cy = (dy1 + dy2) / 2
        # Check if detection center is inside the search zone
        if search_x1 <= cx <= search_x2 and search_y1 <= cy <= search_y2:
            matched.append(det)
            if det["cls"] == with_cls and det["conf"] > best_with_conf:
                best_with_conf = det["conf"]
            if det["cls"] == no_cls and det["conf"] > best_no_conf:
                best_no_conf = det["conf"]

    if DEBUG_HELMET:
        if matched:
            det_str = ", ".join(f"{d['name']}={d['conf']:.2f}" for d in matched)
            print(f"    Rider [{rider_idx}] helmet detections: {det_str}")
        else:
            print(f"    Rider [{rider_idx}] helmet detections: NONE (no detection in search zone)")

    if best_no_conf >= NO_HELMET_MIN_CONF and best_no_conf > best_with_conf:
        return "no_helmet"
    if best_with_conf >= HELMET_CONF:
        return "helmet"
    return "unknown"

# =============================================================================
# Drawing helpers
# =============================================================================

def adaptive_font(img_w, img_h):
    base  = min(img_w, img_h)
    scale = max(0.4, min(1.0, base / 800))
    return scale, max(1, int(scale * 2))

def draw_label(img, text, x, y, bg_color, font_scale, thickness, img_w):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, min(x, img_w - tw - 8))
    y = max(th + bl + 4, y)
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 4, y), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# =============================================================================
# Load models
# =============================================================================

print("Loading detection model (yolo11n) ...")
det_model = YOLO(DETECTION_MODEL_PATH)
print("Detection model ready.")

helmet_model    = None
with_helmet_cls = None
no_helmet_cls   = None

cfg = HELMET_CLASSES[HELMET_MODEL_CHOICE]
if os.path.exists(cfg["path"]):
    print(f"\nLoading helmet model [{HELMET_MODEL_CHOICE}] from {cfg['path']} ...")
    helmet_model    = YOLO(cfg["path"])
    with_helmet_cls = cfg["with_helmet"]
    no_helmet_cls   = cfg["no_helmet"]
    print(f"Helmet model classes: {helmet_model.names}")
    print(f"Using — with_helmet=class {with_helmet_cls}, no_helmet=class {no_helmet_cls}\n")
else:
    print(f"\nWARNING: No helmet model found at '{cfg['path']}' — skipping helmet detection.")
    print(f"  GitHub:      https://github.com/Juliowiwiwiwi/Bike-Helmet-Detction-Model/tree/master/Weights")
    print(f"  HuggingFace: https://huggingface.co/aneesarom/Helmet-Violation-Detection/tree/main")
    print(f"  iam-tsr:     pip install huggingface_hub, then run download_helmet_model.py\n")

# =============================================================================
# Load image
# =============================================================================

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not load image:\n  {IMAGE_PATH}")
    exit()

img_h, img_w = image.shape[:2]
font_scale, font_thickness = adaptive_font(img_w, img_h)
print(f"Image loaded: {img_w}x{img_h}\n")

# =============================================================================
# Stage 1 — Detect bikes and persons
# =============================================================================

results = det_model(image)[0]
bikes, persons = [], []

for box in results.boxes:
    cls  = int(box.cls[0])
    conf = float(box.conf[0])
    if conf < CONF_THRESH:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    if cls in (MOTORCYCLE_CLASS, BICYCLE_CLASS):
        if is_real_motorcycle([x1, y1, x2, y2], img_w, img_h):
            bikes.append({"box": [x1, y1, x2, y2], "conf": conf})
        else:
            print(f"  [filtered] Auto-like box [{x1},{y1},{x2},{y2}] skipped")
    elif cls == PERSON_CLASS:
        persons.append({"box": [x1, y1, x2, y2], "conf": conf})

print(f"Detected {len(bikes)} motorcycle(s), {len(persons)} person(s)\n")

# =============================================================================
# Stage 2 — Run helmet model ONCE on full image, then associate + draw
# =============================================================================

output           = image.copy()
assigned_persons = set()

# Run helmet model once on the full image
# Resize to max 1280px wide for helmet model — it was trained on smaller images
# and returns 0 detections on very large images (3000px+)
helmet_detections = []
if helmet_model is not None:
    h_img = image
    scale = 1.0
    if img_w > 1280:
        scale  = 1280 / img_w
        h_img  = cv2.resize(image, (1280, int(img_h * scale)))
    helmet_detections_raw = run_helmet_on_full_image(helmet_model, h_img)
    # Scale boxes back to original image coords
    for d in helmet_detections_raw:
        x1, y1, x2, y2 = d["box"]
        d["box"] = [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)]
    helmet_detections = helmet_detections_raw
    if DEBUG_HELMET:
        print(f"Helmet model found {len(helmet_detections)} detection(s) in full image:")
        for d in helmet_detections:
            print(f"  {d['name']}={d['conf']:.2f} at {d['box']}")
        print()

for i, bike in enumerate(bikes):
    bx1, by1, bx2, by2 = bike["box"]
    rider_boxes = []

    for j, person in enumerate(persons):
        if j in assigned_persons:
            continue
        if is_rider(person["box"], bike["box"]):
            rider_boxes.append((j, person["box"]))

    rider_count = min(len(rider_boxes), 3)
    for j, _ in rider_boxes:
        assigned_persons.add(j)

    if DEBUG_HELMET and rider_boxes:
        print(f"  Bike [{i}] — {rider_count} rider(s):")

    # Helmet check per rider
    no_helmet_count = 0
    rider_results   = []

    for k, (_, rb) in enumerate(rider_boxes):
        if helmet_model is not None:
            status = match_helmet_to_rider(helmet_detections, rb, with_helmet_cls, no_helmet_cls, k)
        else:
            status = "unknown"
        rider_results.append((rb, status))
        if status == "no_helmet":
            no_helmet_count += 1

    # Violation logic
    triple = rider_count > MAX_RIDERS
    no_hel = no_helmet_count > 0

    if triple and no_hel:
        bike_color = (0, 0, 220)
        label = f"VIOLATION: {rider_count} riders + no helmet"
    elif triple:
        bike_color = (0, 0, 220)
        label = f"VIOLATION: {rider_count} riders"
    elif no_hel:
        bike_color = (0, 80, 255)
        label = f"VIOLATION: no helmet ({no_helmet_count})"
    else:
        bike_color = (255, 120, 0)
        label = f"Riders: {rider_count}"

    # Draw bike box + label
    cv2.rectangle(output, (bx1, by1), (bx2, by2), bike_color, 2)
    draw_label(output, label, bx1, by1, bike_color, font_scale, font_thickness, img_w)

    # Draw rider boxes
    for rb, h_status in rider_results:
        rx1, ry1, rx2, ry2 = rb
        if h_status == "no_helmet":
            rc, rl = (0, 0, 220), "No Helmet!"
        elif h_status == "helmet":
            rc, rl = (0, 200, 0), "Helmet OK"
        else:
            # unknown — don't draw a label, just a neutral box
            rc, rl = (200, 200, 0), "Rider"
        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), rc, 1)
        draw_label(output, rl, rx1, ry1, rc, font_scale * 0.85, font_thickness, img_w)

    # Terminal summary
    parts = []
    if triple: parts.append("TRIPLE RIDING")
    if no_hel: parts.append("NO HELMET")
    if not parts: parts.append("OK")
    print(f"  Bike [{i}]  conf={bike['conf']:.2f}  riders={rider_count}  {' + '.join(parts)}\n")

# =============================================================================
# Show result
# =============================================================================

display_w = max(img_w, 1200)
display_h = int(img_h * (display_w / img_w))
display   = cv2.resize(output, (display_w, display_h))

cv2.namedWindow("Traffic Violation Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Violation Detection", min(display_w, 1600), min(display_h, 900))
cv2.imshow("Traffic Violation Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
