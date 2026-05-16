from ultralytics import YOLO
import cv2

# -----------------------------------
# Config
# -----------------------------------

DETECTION_MODEL_PATH = "models/yolo11n.pt"
HELMET_MODEL_PATH    = "models/helmet.pt"       # your helmet model goes here
IMAGE_PATH           = r"C:\Users\kaush\Downloads\Traffic-Violation-Detection\Images to test on\pexels-mahmutyilmaz-35317056.jpg.jpeg"

CONF_THRESH   = 0.30
MAX_RIDERS    = 2

# COCO class IDs
PERSON_CLASS     = 0
BICYCLE_CLASS    = 1
MOTORCYCLE_CLASS = 3

# Rider association: fraction of PERSON box that must overlap BIKE box
OVERLAP_THRESH = 0.25

# Motorcycle size filter (filters out autos misdetected as motorcycles)
MAX_BIKE_WIDTH_FRACTION = 0.25   # bike can't be wider than 25% of image
MAX_ASPECT_RATIO        = 1.4    # width/height ratio — autos are wide, bikes are tall

# Helmet model class IDs (update these once you know your helmet model's classes)
HELMET_CLASS    = 0   # "helmet" or "with_helmet"
NO_HELMET_CLASS = 1   # "no_helmet" or "without_helmet"
HELMET_CONF     = 0.30

# -----------------------------------
# Motorcycle filter
# -----------------------------------

def is_real_motorcycle(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0:
        return False
    aspect = bw / bh
    width_fraction = bw / img_w
    if aspect > MAX_ASPECT_RATIO:
        return False
    if width_fraction > MAX_BIKE_WIDTH_FRACTION:
        return False
    return True

# -----------------------------------
# Rider association
# -----------------------------------

def box_overlap_ratio(person_box, bike_box):
    """Fraction of PERSON box that overlaps with BIKE box."""
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

    overlap_area = (ov_x2 - ov_x1) * (ov_y2 - ov_y1)
    person_area  = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return 0.0
    return overlap_area / person_area


def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    # Rule 1: significant box overlap
    if box_overlap_ratio(person_box, bike_box) >= OVERLAP_THRESH:
        return True

    # Rule 2: center-x inside bike AND person bottom near bike vertically
    center_x      = (px1 + px2) / 2
    person_bottom = py2
    bike_height   = by2 - by1

    if bx1 <= center_x <= bx2 and by1 - bike_height * 0.5 <= person_bottom <= by2 + 20:
        return True

    return False

# -----------------------------------
# Helmet detection
# -----------------------------------

def check_helmet(helmet_model, image, person_box):
    """
    Crop the HEAD region of a rider and run the helmet model on it.
    Returns: "helmet", "no_helmet", or "unknown"
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1

    # Crop top 40% of person box = head/helmet region
    head_y2 = py1 + int(ph * 0.40)
    head_x1 = max(0, px1)
    head_y1 = max(0, py1)
    head_x2 = px2
    head_y2 = min(image.shape[0], head_y2)

    if head_x2 <= head_x1 or head_y2 <= head_y1:
        return "unknown"

    head_crop = image[head_y1:head_y2, head_x1:head_x2]
    if head_crop.size == 0:
        return "unknown"

    results = helmet_model(head_crop, conf=HELMET_CONF, verbose=False)

    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < HELMET_CONF:
                continue
            if cls == HELMET_CLASS:
                return "helmet"
            elif cls == NO_HELMET_CLASS:
                return "no_helmet"

    return "unknown"

# -----------------------------------
# Drawing helpers
# -----------------------------------

def adaptive_font(img_w, img_h):
    base = min(img_w, img_h)
    scale = max(0.4, min(1.0, base / 800))
    thickness = max(1, int(scale * 2))
    return scale, thickness


def draw_label(img, text, x, y, bg_color, font_scale, thickness, img_w=None):
    """Draw text with solid background, clamped to image edges."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    # Clamp x so label doesn't go off right edge
    if img_w is not None:
        x = min(x, img_w - tw - 8)
    x = max(x, 0)
    # Clamp y so label doesn't go above top edge
    y = max(y, th + baseline + 4)

    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y), bg_color, -1)
    cv2.putText(
        img, text,
        (x + 2, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), thickness
    )

# -----------------------------------
# Load models
# -----------------------------------

print("Loading detection model (yolo11n) ...")
det_model = YOLO(DETECTION_MODEL_PATH)

# Load helmet model if it exists, otherwise skip helmet detection
import os
helmet_model = None
if os.path.exists(HELMET_MODEL_PATH):
    print("Loading helmet model ...")
    helmet_model = YOLO(HELMET_MODEL_PATH)
    print("Helmet model loaded.")
else:
    print(f"Helmet model not found at '{HELMET_MODEL_PATH}' — skipping helmet detection.")

print("Models ready.\n")

# -----------------------------------
# Load image
# -----------------------------------

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not load image:\n  {IMAGE_PATH}")
    exit()

img_h, img_w = image.shape[:2]
font_scale, font_thickness = adaptive_font(img_w, img_h)

# -----------------------------------
# Stage 1: Detect bikes and persons
# -----------------------------------

results = det_model(image)[0]

bikes   = []
persons = []

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

# -----------------------------------
# Stage 2: Associate riders + helmet check
# -----------------------------------

output = image.copy()
assigned_persons = set()

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

    # Helmet check for each rider
    no_helmet_count = 0
    rider_helmet_status = []

    for _, rb in rider_boxes:
        if helmet_model is not None:
            status = check_helmet(helmet_model, image, rb)
        else:
            status = "unknown"
        rider_helmet_status.append((rb, status))
        if status == "no_helmet":
            no_helmet_count += 1

    # Violation: too many riders OR any rider without helmet
    triple_riding = rider_count > MAX_RIDERS
    helmet_violation = no_helmet_count > 0

    if triple_riding and helmet_violation:
        bike_color = (0, 0, 255)        # red
        label = f"VIOLATION: {rider_count} riders + no helmet"
    elif triple_riding:
        bike_color = (0, 0, 255)        # red
        label = f"VIOLATION: {rider_count} riders"
    elif helmet_violation:
        bike_color = (0, 140, 255)      # orange-red
        label = f"VIOLATION: no helmet ({no_helmet_count} rider(s))"
    else:
        bike_color = (255, 120, 0)      # orange
        label = f"Riders: {rider_count}"

    # Draw bike box
    cv2.rectangle(output, (bx1, by1), (bx2, by2), bike_color, 2)

    # Draw rider boxes with helmet status
    for rb, h_status in rider_helmet_status:
        rx1, ry1, rx2, ry2 = rb

        if h_status == "no_helmet":
            rider_color  = (0, 0, 255)
            rider_label  = "No Helmet!"
        elif h_status == "helmet":
            rider_color  = (0, 220, 0)
            rider_label  = "Helmet OK"
        else:
            rider_color  = (0, 220, 0)
            rider_label  = "Rider"

        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), rider_color, 1)
        draw_label(output, rider_label, rx1, ry1, rider_color,
                   font_scale * 0.85, font_thickness, img_w)

    # Draw bike label
    draw_label(output, label, bx1, by1, bike_color, font_scale, font_thickness, img_w)

    status_str = []
    if triple_riding:   status_str.append("TRIPLE RIDING")
    if helmet_violation: status_str.append("NO HELMET")
    if not status_str:  status_str.append("OK")
    print(f"  Bike [{i}]  conf={bike['conf']:.2f}  riders={rider_count}  {' + '.join(status_str)}")

# -----------------------------------
# Show result
# -----------------------------------

display_w = max(img_w, 1200)
display_h = int(img_h * (display_w / img_w))
display   = cv2.resize(output, (display_w, display_h))

cv2.namedWindow("Traffic Violation Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Violation Detection", min(display_w, 1600), min(display_h, 900))
cv2.imshow("Traffic Violation Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
