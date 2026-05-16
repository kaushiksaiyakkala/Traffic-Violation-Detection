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

def run_helmet_on_full_image(helmet_model, image, model_name=""):
    """
    Run one helmet model on the full image.
    Returns list of dicts: {box, cls, conf, name, model}
    """
    results = helmet_model(image, conf=0.20, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = helmet_model.names.get(cls, str(cls))
        detections.append({"box": [x1, y1, x2, y2], "cls": cls, "conf": conf, "name": name, "model": model_name})
    return detections


def run_helmet_ensemble(helmet_models_list, image, img_w, img_h):
    """
    Run ALL helmet models and merge detections.
    Resizes image if needed, scales boxes back to original coords.
    Returns combined detection list from all models.
    """
    all_detections = []
    scale = 1.0
    h_img = image
    if img_w > 1280:
        scale = 1280 / img_w
        h_img = cv2.resize(image, (1280, int(img_h * scale)))

    for hcfg in helmet_models_list:
        dets = run_helmet_on_full_image(hcfg["model"], h_img, hcfg["name"])
        # Scale boxes back to original coords
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            d["box"] = [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)]
            # Remap cls to standard: with_helmet=0, no_helmet=1
            d["std_cls"] = "with_helmet" if d["cls"] == hcfg["with_helmet"] else "no_helmet"
        all_detections.extend(dets)

    return all_detections


def match_helmet_to_rider(helmet_detections, person_box, with_cls, no_cls, rider_idx):
    """
    Find helmet detections whose center falls inside the person box
    (expanded upward 60% to cover head region above the person box).
    Returns: "helmet", "no_helmet", or "unknown"
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1

    # Search zone: expand upward 100% and sideways 80%
    # Helmet detections are often offset from the person box center
    # so we need a generous horizontal margin
    search_x1 = max(0, px1 - int(pw * 0.40))
    search_y1 = max(0, py1 - int(ph * 1.00))
    search_x2 = px2 + int(pw * 0.40)
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
            if det.get("std_cls", "") == "with_helmet" and det["conf"] > best_with_conf:
                best_with_conf = det["conf"]
            if det.get("std_cls", "") == "no_helmet" and det["conf"] > best_no_conf:
                best_no_conf = det["conf"]

    if DEBUG_HELMET:
        if matched:
            det_str = ", ".join(f"{d['name']}={d['conf']:.2f}" for d in matched)
            print(f"    Rider [{rider_idx}] helmet detections: {det_str}")
        else:
            print(f"    Rider [{rider_idx}] helmet detections: NONE | person={person_box} search=[{search_x1},{search_y1},{search_x2},{search_y2}]")

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

# Load all available helmet models — we run both and merge detections
# This gives better coverage than any single model alone
HELMET_MODEL_CONFIGS = [
    {"path": "models/helmet_github.pt",  "with_helmet": 0, "no_helmet": 1, "name": "github"},
    {"path": "models/helmet_iamtsr.pt",  "with_helmet": 0, "no_helmet": 1, "name": "iam-tsr"},
]

helmet_models = []
for hcfg in HELMET_MODEL_CONFIGS:
    if os.path.exists(hcfg["path"]):
        print(f"Loading helmet model [{hcfg['name']}] ...")
        m = YOLO(hcfg["path"])
        helmet_models.append({**hcfg, "model": m})
        print(f"  Classes: {m.names}")
    else:
        print(f"Skipping [{hcfg['name']}] — not found at {hcfg['path']}")

# Keep single model vars for backward compat (used in match function)
with_helmet_cls = 0
no_helmet_cls   = 1

if not helmet_models:
    print("WARNING: No helmet models found — skipping helmet detection.")
else:
    print(f"\nUsing {len(helmet_models)} helmet model(s) in ensemble.\n")

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

# Run ALL helmet models on full image and merge detections (ensemble)
helmet_detections = []
if helmet_models:
    helmet_detections = run_helmet_ensemble(helmet_models, image, img_w, img_h)
    if DEBUG_HELMET:
        print(f"Ensemble found {len(helmet_detections)} total detection(s):")
        for d in helmet_detections:
            print(f"  [{d['model']}] {d['name']}={d['conf']:.2f} std={d['std_cls']} at {d['box']}")
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
        if helmet_models:
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
