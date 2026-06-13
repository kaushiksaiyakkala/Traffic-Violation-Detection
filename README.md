# Traffic Violation Detection

A computer vision pipeline that analyzes street-scene images of two-wheelers (motorcycles/bicycles) and automatically detects common traffic violations: **triple riding** (more than two riders on one vehicle) and **riding without a helmet**. For each detected violation, the system attempts to extract the vehicle's license plate using OCR.

## Features

- **Vehicle & rider detection** — YOLOv11-Large detects motorcycles, bicycles, and persons in the scene.
- **Rider-to-vehicle association** — Hungarian algorithm (with a greedy fallback) assigns detected persons to the correct bike based on spatial overlap and eligibility heuristics.
- **Helmet classification** — An ensemble of two helmet-detection models (with fallback matching against the bike bounding box) classifies each rider as wearing a helmet or not.
- **License plate recognition** — A dedicated YOLO plate-detector locates the plate on flagged vehicles, which is then upscaled, preprocessed (CLAHE, bilateral filtering), and read using PaddleOCR.
- **Violation reporting** — For every flagged vehicle, outputs the violation type(s), rider count, helmet violation count, and recognized plate text.

## Pipeline Overview

1. **Detection Stage** — Run YOLOv11-L on the (resized) input image to detect persons, motorcycles, and bicycles.
2. **Association Stage** — Match riders to bikes using overlap ratio and eligibility rules (area, aspect ratio, center-of-mass containment), solved via Hungarian assignment.
3. **Helmet Stage** — Run the helmet model ensemble over the image, then match each detection to a rider using a positional search zone around the rider's head, with a bike-box fallback if no direct match is found.
4. **Violation Check** — Flag a vehicle if rider count > 2 (triple riding) or any rider is classified `no_helmet`.
5. **Plate Stage** — For flagged vehicles, crop and upscale the bike region, detect the plate, preprocess it, and run OCR to extract the registration number.

## Requirements

- Python 3.10+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`ultralytics`)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (`paddleocr`)
- OpenCV (`opencv-python`)
- NumPy, SciPy

```bash
pip install ultralytics paddleocr opencv-python numpy scipy
```

> **Note (Windows + GPU users):** If `torch.cuda.is_available()` returns `False`, reinstall PyTorch with CUDA support, e.g.:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

## Models

Place the following model weights inside the `models/` directory:

| File | Purpose |
|---|---|
| `yolo11l.pt` | Vehicle & person detection |
| `helmet_github.pt` | Helmet classification (ensemble member 1) |
| `helmet_iamtsr.pt` | Helmet classification (ensemble member 2) |
| `best.pt` | License plate detection |
| `paddleocr/det`, `paddleocr/rec`, `paddleocr/cls` | PaddleOCR model directories |

## Usage

```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result = detector.predict("test.jpg")

print(result)
```

### Example Output

```python
{
  "violations": [
    {
      "violation_types": ["triple_riding", "no_helmet"],
      "num_riders": 3,
      "helmet_violations": 1,
      "license_plate": "KA01AB1234"
    }
  ]
}
```

If no violations are detected, `violations` is an empty list.

## Configuration

Key tunable thresholds are defined as constants at the top of `solution.py`:

| Constant | Default | Description |
|---|---|---|
| `CONF_THRESH` | `0.30` | Minimum confidence for vehicle/person detections |
| `MAX_RIDERS` | `2` | Maximum allowed riders before flagging triple riding |
| `OVERLAP_THRESH` | `0.15` | Minimum overlap ratio for rider-bike association |
| `HELMET_CONF` | `0.35` | Minimum confidence to confirm "with helmet" |
| `NO_HELMET_MIN_CONF` | `0.50` | Minimum confidence to confirm "no helmet" |
| `PLATE_CONF` | `0.15` | Minimum confidence for plate detection |
| `DETECT_UPSCALE` | `2.0` | Upscale factor applied to bike crop before plate detection |
| `USE_HUNGARIAN` | `True` | Use Hungarian assignment (vs. greedy) for rider-bike matching |
| `DEBUG_HELMET` | `False` | Print per-rider helmet matching diagnostics |

## Project Structure

```
Traffic-Violation-Detection/
├── solution.py          # Main detector class and inference pipeline
├── models/               # YOLO + PaddleOCR model weights (not included)
└── images/               # Sample test images
```

## Notes

- Input images are resized to a target width of 1280px (or upscaled if smaller than 400x300) before detection to balance speed and accuracy.
- The helmet matching logic first searches a region above and around each rider's bounding box; if no helmet detection is found there, it falls back to matching detections within the bike's bounding box.
- OCR results below a confidence of `0.30` per text line, or with fewer than 4 alphanumeric characters overall, are discarded.
