from ultralytics import YOLO
import cv2

# -----------------------------------
# Load YOLO11 model
# -----------------------------------

model = YOLO("models/yolo11n.pt")

# -----------------------------------
# Load image
# -----------------------------------

image_path = "test6.jpeg"

image = cv2.imread(image_path)

# Safety check
if image is None:
    print("Error: Could not load image")
    exit()

# -----------------------------------
# Run inference
# -----------------------------------

results = model(image)[0]

# -----------------------------------
# Store detections
# -----------------------------------

motorcycles = []
persons = []

# COCO class IDs
PERSON_CLASS = 0
MOTORCYCLE_CLASS = 3

# -----------------------------------
# Detect motorcycles and persons
# -----------------------------------

for box in results.boxes:

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    if conf < 0.3:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Motorcycle
    if cls == MOTORCYCLE_CLASS:

        motorcycles.append({
            "box": [x1, y1, x2, y2],
            "conf": conf
        })

    # Person
    elif cls == PERSON_CLASS:

        persons.append({
            "box": [x1, y1, x2, y2],
            "conf": conf
        })

# -----------------------------------
# Output image
# -----------------------------------

output_image = image.copy()

# -----------------------------------
# Process each motorcycle
# -----------------------------------

for bike in motorcycles:

    bx1, by1, bx2, by2 = bike["box"]

    rider_count = 0

    # Draw motorcycle box
    cv2.rectangle(
        output_image,
        (bx1, by1),
        (bx2, by2),
        (255, 0, 0),
        3
    )

    # -----------------------------------
    # Count riders near motorcycle
    # -----------------------------------

    for person in persons:

        px1, py1, px2, py2 = person["box"]

        # Person center point
        center_x = (px1 + px2) // 2
        center_y = (py1 + py2) // 2

        # Association logic
        if (
            bx1 - 50 <= center_x <= bx2 + 50 and
            by1 - 100 <= center_y <= by2 + 100
        ):

            rider_count += 1

            # Draw rider box
            cv2.rectangle(
                output_image,
                (px1, py1),
                (px2, py2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_image,
                "Rider",
                (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------------
    # Violation logic
    # -----------------------------------

    if rider_count > 2:

        violation_text = f"VIOLATION: {rider_count} Riders"

        color = (0, 0, 255)  # Red

    else:

        violation_text = f"Riders: {rider_count}"

        color = (255, 255, 0)  # Yellow

    # Display count
    cv2.putText(
        output_image,
        violation_text,
        (bx1, by1 - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3
    )

# -----------------------------------
# Resize for large display
# -----------------------------------

display_image = cv2.resize(
    output_image,
    (1600, 900)
)

# -----------------------------------
# Create popup window
# -----------------------------------

cv2.namedWindow(
    "Bike Rider Violation Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Bike Rider Violation Detection",
    1600,
    900
)

# -----------------------------------
# Show output
# -----------------------------------

cv2.imshow(
    "Bike Rider Violation Detection",
    display_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
