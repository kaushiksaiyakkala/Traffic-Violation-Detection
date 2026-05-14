from ultralytics import YOLO
import cv2

# Load YOLO11 model
model = YOLO("models/yolo11n.pt")

# Load image
image_path = "test.jpeg"

image = cv2.imread(image_path)

# Safety check
if image is None:
    print("Error: Could not read image")
    exit()

# Run inference
results = model(image)[0]

# Loop through detections
for box in results.boxes:

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    # Ignore low confidence detections
    if conf < 0.3:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # COCO class IDs
    # person = 0
    # motorcycle = 3

    if cls == 0:
        label = "Person"

    elif cls == 3:
        label = "Motorcycle"

    else:
        continue

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw label
    cv2.putText(
        image,
        f"{label} {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Resize window for better viewing
display_image = cv2.resize(image, (1200, 800))

# Create popup window
cv2.imshow("YOLO11 Detection", display_image)

# Wait until key press
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()
