from ultralytics import YOLO
from PIL import Image
import base64
import io
import json
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")
model = YOLO(model_path)

def handler(request):
    try:
        body = request.json()

        if "image" not in body:
            return {
                "statusCode": 400,
                "body": "Missing image"
            }

        # decode base64 → image
        image_bytes = base64.b64decode(body["image"])
        img = Image.open(io.BytesIO(image_bytes))

        results = model(img)

        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist()
            })

        return {
            "statusCode": 200,
            "body": json.dumps({"detections": detections})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
