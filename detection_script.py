import torch
import cv2
import os
import json
from pathlib import Path

# Load YOLOv5 model (make sure yolov5 repo is cloned inside project)
MODEL_PATH = "models/best_glove.pt"
OUTPUT_DIR = "output"
LOG_DIR = "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load model
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")

def run_detection(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    results.save(save_dir=OUTPUT_DIR)

    # Save JSON logs
    log_data = results.pandas().xyxy[0].to_dict(orient="records")
    log_path = os.path.join(LOG_DIR, Path(image_path).stem + ".json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Processed: {image_path}")
    return output_path, log_path


if __name__ == "__main__":
    test_images = list(Path("data/images/test").glob("*.jpg")) + list(Path("data/images/test").glob("*.png"))
    for img_path in test_images:
        run_detection(str(img_path))
