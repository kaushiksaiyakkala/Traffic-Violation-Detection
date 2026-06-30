from solution_visualize import TrafficViolationDetector

image_path = r"C:\Users\kaush\Downloads\Traffic-Violation-Detection\Images to test on\WhatsApp Image 2026-05-16 at 10.51.01 PM.jpeg"

print("Loading detector...")
detector = TrafficViolationDetector(model_dir="./models")

print("Running prediction...")
result = detector.predict(
    image_path,
    visualize=True,
    output_path="output.jpg"
)

print(result)
print("Saved output.jpg")
print("Done")