from ultralytics import YOLO
import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

# =============================================================================
# CONFIG
# =============================================================================

IMAGE_PATH = r"C:\Users\kaush\Downloads\Traffic-Violation-Detection\Images to test on\WhatsApp Image 2026-05-14 at 8.41.08 PM.jpeg"
DEBUG_HELMET   = True
USE_HUNGARIAN  = False  # True = Hungarian matching, False = Greedy

# =============================================================================
# Fixed config
# =============================================================================

DETECTION_MODEL_PATH    = "models/yolo11l.pt"
CONF_THRESH             = 0.30
MAX_RIDERS              = 2

PERSON_CLASS            = 0
BICYCLE_CLASS           = 1
MOTORCYCLE_CLASS        = 3

OVERLAP_THRESH          = 0.15
MAX_BIKE_WIDTH_FRACTION = 0.22
MAX_ASPECT_RATIO        = 1.8

HELMET_CONF             = 0.35   # min conf to confirm "Helmet OK"
NO_HELMET_MIN_CONF      = 0.50   # min conf to flag a violation

# =============================================================================
# Motorcycle filter
# =============================================================================

def is_real_motorcycle(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0: return False
    aspect = bw / bh
    width_pct = bw / img_w

    # Autos are wide relative to image AND have moderate-to-wide aspect
    # Close-up bikes can be very wide in absolute terms but rare
    if aspect > 1.25 and width_pct > 0.35: return False
    if width_pct > 0.55: return False  # nothing is this wide except autos

    return True

# =============================================================================
# Rider association helpers
# =============================================================================

def box_overlap_ratio(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    ov_y1 = max(py1, by1); ov_y2 = min(py2, by2)
    if ov_y2 <= ov_y1: return 0.0
    ov_x1 = max(px1, bx1); ov_x2 = min(px2, bx2)
    if ov_x2 <= ov_x1: return 0.0
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0: return 0.0
    return ((ov_x2 - ov_x1) * (ov_y2 - ov_y1)) / person_area

def is_eligible(person_box, bike_box, img_w=None, img_h=None):
    """Check if person is eligible to be a rider on this bike."""
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    # Filter out very large person boxes — foreground pedestrians, not riders
    if img_w and img_h:
        person_area = (px2 - px1) * (py2 - py1)
        image_area  = img_w * img_h
        if person_area > 0.12 * image_area:  # skip if >12% of image
            return False

    # If person box is much wider than bike box, they're likely not a rider
    # (auto passenger boxes are wide, bike rider boxes are narrow)
    person_w = px2 - px1
    bike_w   = bx2 - bx1
    if bike_w > 0 and person_w > bike_w * 2.5:
        return False

    if box_overlap_ratio(person_box, bike_box) >= OVERLAP_THRESH:
        return True

    # Center rule — person center must be inside bike box horizontally
    # AND person bottom must overlap meaningfully with bike vertically
    center_x = (px1 + px2) / 2
    bike_h   = by2 - by1
    if bx1 <= center_x <= bx2 and by1 + bike_h * 0.10 <= py2 <= by2 + 10:
        return True
    return False

# =============================================================================
# Hungarian matching
# Person-centric: each person assigned to best bike, multiple persons per bike OK
# =============================================================================

def hungarian_assign(bikes, persons, img_w=None, img_h=None):
    """
    Assign each person to at most one bike using Hungarian algorithm.
    Multiple persons can be assigned to the same bike.
    Returns dict: {bike_idx: [person_idx, ...]}
    """
    assignment = {i: [] for i in range(len(bikes))}
    if not bikes or not persons:
        return assignment

    n_bikes   = len(bikes)
    n_persons = len(persons)

    # Build cost matrix: rows=persons, cols=bikes
    # We solve person→bike assignment (each person to one bike)
    cost = np.full((n_persons, n_bikes), 1000.0)
    for j, person in enumerate(persons):
        for i, bike in enumerate(bikes):
            overlap = box_overlap_ratio(person["box"], bike["box"])
            if is_eligible(person["box"], bike["box"]):
                # Lower cost = better match; prefer higher overlap
                cost[j, i] = 1.0 - overlap

    # Run Hungarian on person→bike
    person_indices, bike_indices = linear_sum_assignment(cost)

    assigned = set()
    for pi, bi in zip(person_indices, bike_indices):
        if cost[pi, bi] < 1000.0:
            assignment[bi].append(pi)
            assigned.add(pi)

    # Greedy pass: assign remaining eligible persons to their best bike
    for pi, person in enumerate(persons):
        if pi in assigned:
            continue
        best_bi    = None
        best_score = -1
        for bi, bike in enumerate(bikes):
            overlap = box_overlap_ratio(person["box"], bike["box"])
            if is_eligible(person["box"], bike["box"], img_w, img_h) and overlap > best_score:
                best_score = overlap
                best_bi    = bi
        if best_bi is not None:
            assignment[best_bi].append(pi)
            assigned.add(pi)

    return assignment

# =============================================================================
# Greedy assignment
# =============================================================================

def greedy_assign(bikes, persons, img_w=None, img_h=None):
    """
    Greedy assignment — assigns each person to the bike with highest overlap.
    Processes pairs in descending overlap order so best matches go first.
    Each person assigned to at most one bike, multiple persons per bike OK.
    """
    assignment = {i: [] for i in range(len(bikes))}
    assigned   = set()

    # Score every eligible person-bike pair
    pairs = []
    for pi, person in enumerate(persons):
        for bi, bike in enumerate(bikes):
            if not is_eligible(person["box"], bike["box"], img_w, img_h):
                continue
            overlap = box_overlap_ratio(person["box"], bike["box"])
            pairs.append((overlap, pi, bi))

    # Sort by overlap descending — best matches assigned first
    pairs.sort(reverse=True)

    for overlap, pi, bi in pairs:
        if pi in assigned:
            # Already assigned — but allow shared assignment if overlap is high
            # (person genuinely on both bikes e.g. side by side)
            if overlap >= 0.30:
                assignment[bi].append(pi)
            continue
        assignment[bi].append(pi)
        assigned.add(pi)

    return assignment

# =============================================================================
# Helmet detection
# =============================================================================

def run_helmet_on_full_image(helmet_model, image, model_name=""):
    results = helmet_model(image, conf=0.20, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = helmet_model.names.get(cls, str(cls))
        detections.append({"box": [x1, y1, x2, y2], "cls": cls,
                            "conf": conf, "name": name, "model": model_name})
    return detections


def run_helmet_ensemble(helmet_models_list, image, img_w, img_h):
    all_detections = []
    scale = 1.0
    h_img = image
    if img_w > 1280:
        scale = 1280 / img_w
        h_img = cv2.resize(image, (1280, int(img_h * scale)))

    for hcfg in helmet_models_list:
        dets = run_helmet_on_full_image(hcfg["model"], h_img, hcfg["name"])
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            d["box"]     = [int(x1/scale), int(y1/scale),
                             int(x2/scale), int(y2/scale)]
            d["std_cls"] = "with_helmet" if d["cls"] == hcfg["with_helmet"] else "no_helmet"
        all_detections.extend(dets)

    return all_detections


def match_helmet_to_rider(helmet_detections, person_box, bike_box, with_cls, no_cls, rider_idx):
    """
    Match helmet detection to rider using proximity-weighted scoring.
    Closer detections beat farther ones even if lower confidence.
    Falls back to full bike box if person box search finds nothing.
    Returns: "helmet", "no_helmet", or "unknown"
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1

    head_cx = (px1 + px2) / 2
    head_cy = py1 + int(ph * 0.15)

    # Search zone around person box
    search_x1 = max(0, px1 - int(pw * 0.50))
    search_y1 = max(0, py1 - int(ph * 1.00))
    search_x2 = px2 + int(pw * 0.50)
    search_y2 = py2

    best_with_score = 0.0
    best_no_score   = 0.0
    matched = []

    for det in helmet_detections:
        dx1, dy1, dx2, dy2 = det["box"]
        cx = (dx1 + dx2) / 2
        cy = (dy1 + dy2) / 2

        if not (search_x1 <= cx <= search_x2 and search_y1 <= cy <= search_y2):
            continue

        # Proximity-weighted score: closer = higher score
        dist  = ((cx - head_cx)**2 + (cy - head_cy)**2) ** 0.5
        score = det["conf"] / (1.0 + dist / max(pw, 1))

        matched.append({**det, "score": score})
        if det["std_cls"] == "with_helmet" and score > best_with_score:
            best_with_score = score
        if det["std_cls"] == "no_helmet" and score > best_no_score:
            best_no_score = score

    if DEBUG_HELMET:
        if matched:
            det_str = ", ".join(f"{d['name']}={d['conf']:.2f}(s={d['score']:.2f})" for d in matched)
            print(f"    Rider [{rider_idx}] detections: {det_str}")
        else:
            print(f"    Rider [{rider_idx}] NONE | person={person_box} search=[{search_x1},{search_y1},{search_x2},{search_y2}]")

    # Decision using proximity-weighted scores
    if best_no_score >= NO_HELMET_MIN_CONF and best_no_score > best_with_score:
        return "no_helmet"
    if best_with_score >= HELMET_CONF:
        return "helmet"
    if matched:
        # Borderline — decide based on which class scored higher
        if best_with_score >= best_no_score:
            return "helmet"   # model leans toward helmet
        else:
            return "no_helmet"  # model leans toward no helmet

    # Fallback: search within full bike box
    if bike_box is not None:
        bx1, by1, bx2, by2 = bike_box
        fb_with = fb_no = 0.0
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
            det_str = ", ".join(f"{d['name']}={d['conf']:.2f}" for d in fb_matched)
            print(f"    Rider [{rider_idx}] bike-box fallback: {det_str}")
        if fb_no >= NO_HELMET_MIN_CONF and fb_no > fb_with:
            return "no_helmet"
        if fb_with >= HELMET_CONF:
            return "helmet"
        if fb_matched:
            if fb_with >= fb_no:
                return "helmet"
            else:
                return "no_helmet"

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

print("Loading detection model ...")
det_model = YOLO(DETECTION_MODEL_PATH)
print("Detection model ready.")

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
        print(f"Skipping [{hcfg['name']}] — not found")

with_helmet_cls = 0
no_helmet_cls   = 1

if not helmet_models:
    print("WARNING: No helmet models found.")
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

# Use lower confidence on small images — large model is conservative on low-res
effective_conf = CONF_THRESH
det_image = image
if img_w < 400 or img_h < 300:
    effective_conf = 0.20
    # Upscale small images to give model more detail to work with
    scale = max(640 / img_w, 640 / img_h)
    det_image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
    print(f"Small image — upscaled to {det_image.shape[1]}x{det_image.shape[0]}, conf={effective_conf}")

results = det_model(det_image, conf=effective_conf)[0]

# Scale boxes back to original image coords if upscaled
det_scale_x = img_w / det_image.shape[1] if det_image is not image else 1.0
det_scale_y = img_h / det_image.shape[0] if det_image is not image else 1.0
bikes, persons = [], []

for box in results.boxes:
    cls  = int(box.cls[0])
    conf = float(box.conf[0])
    if conf < CONF_THRESH:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = int(x1 * det_scale_x); y1 = int(y1 * det_scale_y)
    x2 = int(x2 * det_scale_x); y2 = int(y2 * det_scale_y)
    if cls in (MOTORCYCLE_CLASS, BICYCLE_CLASS):
        if is_real_motorcycle([x1, y1, x2, y2], img_w, img_h):
            bikes.append({"box": [x1, y1, x2, y2], "conf": conf})
        else:
            print(f"  [filtered] Auto-like box [{x1},{y1},{x2},{y2}] skipped")
    elif cls == PERSON_CLASS:
        persons.append({"box": [x1, y1, x2, y2], "conf": conf})

print(f"Detected {len(bikes)} motorcycle(s), {len(persons)} person(s)\n")

# Debug eligible pairs
print("Eligible person-bike pairs:")
for pi, person in enumerate(persons):
    for bi, bike in enumerate(bikes):
        if is_eligible(person["box"], bike["box"], img_w, img_h):
            overlap = box_overlap_ratio(person["box"], bike["box"])
            print(f"  Person[{pi}] {person['box']} -> Bike[{bi}] {bike['box']} overlap={overlap:.3f}")
print()

# =============================================================================
# Stage 2 — Helmet ensemble on full image
# =============================================================================

output = image.copy()

helmet_detections = []
if helmet_models:
    helmet_detections = run_helmet_ensemble(helmet_models, image, img_w, img_h)
    if DEBUG_HELMET:
        print(f"Ensemble found {len(helmet_detections)} total detection(s):")
        for d in helmet_detections:
            print(f"  [{d['model']}] {d['name']}={d['conf']:.2f} std={d['std_cls']} at {d['box']}")
        print()

# =============================================================================
# Stage 3 — Hungarian assignment + violation detection
# =============================================================================

if USE_HUNGARIAN:
    assignment = hungarian_assign(bikes, persons, img_w, img_h)
else:
    assignment = greedy_assign(bikes, persons, img_w, img_h)

for i, bike in enumerate(bikes):
    bx1, by1, bx2, by2 = bike["box"]
    person_indices = assignment[i]
    rider_boxes    = [(j, persons[j]["box"]) for j in person_indices]
    rider_count    = min(len(rider_boxes), 3)

    if DEBUG_HELMET and rider_boxes:
        print(f"  Bike [{i}] — {rider_count} rider(s):")

    # Helmet check per rider
    no_helmet_count = 0
    rider_results   = []

    for k, (_, rb) in enumerate(rider_boxes):
        if helmet_models:
            status = match_helmet_to_rider(
                helmet_detections, rb, [bx1, by1, bx2, by2],
                with_helmet_cls, no_helmet_cls, k
            )
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

    cv2.rectangle(output, (bx1, by1), (bx2, by2), bike_color, 2)
    draw_label(output, label, bx1, by1, bike_color, font_scale, font_thickness, img_w)

    for rb, h_status in rider_results:
        rx1, ry1, rx2, ry2 = rb
        if h_status == "no_helmet":
            rc, rl = (0, 0, 220), "No Helmet!"
        elif h_status == "helmet":
            rc, rl = (0, 200, 0), "Helmet OK"
        else:
            rc, rl = (180, 180, 0), "Helmet: Unknown"
        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), rc, 1)
        draw_label(output, rl, rx1, ry1, rc, font_scale * 0.85, font_thickness, img_w)

    parts = []
    if triple: parts.append("TRIPLE RIDING")
    if no_hel: parts.append("NO HELMET")
    if not parts: parts.append("OK")
    print(f"  Bike [{i}]  conf={bike['conf']:.2f}  riders={rider_count}  {' + '.join(parts)}\n")

# =============================================================================
# Save + show
# =============================================================================

display_w = max(img_w, 1200)
display_h = int(img_h * (display_w / img_w))
display   = cv2.resize(output, (display_w, display_h))

cv2.imwrite("output.jpg", display)
print(f"\nSaved output.jpg")

try:
    cv2.namedWindow("Traffic Violation Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Violation Detection", min(display_w, 1600), min(display_h, 900))
    cv2.imshow("Traffic Violation Detection", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    print("Open output.jpg to view results")
