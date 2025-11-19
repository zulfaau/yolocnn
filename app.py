from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# =======================================================
# 🔥 LAZY LOAD — Model hanya diload saat pertama dipakai
# =======================================================
YOLO_MODEL = None

def load_yolo():
    global YOLO_MODEL
    
    if YOLO_MODEL is not None:
        return YOLO_MODEL
    
    try:
        # Prefer load dari file lokal (HuggingFace tidak download ulang)
        if os.path.exists("yolov8n.pt"):
            print("🚀 Loading YOLO locally...")
            YOLO_MODEL = YOLO("yolov8n.pt")
        else:
            print("⬇️ Downloading YOLO from Ultralytics...")
            YOLO_MODEL = YOLO("yolov8n.pt")  # auto-download 6MB
        print("✅ YOLO Ready!")
    except Exception as e:
        print("❌ YOLO gagal load:", e)
        YOLO_MODEL = None
    
    return YOLO_MODEL


# =======================================================
# 🔥 HOME PAGE
# =======================================================
@app.route("/")
def home():
    return render_template("object_detection.html")


# =======================================================
# 🔥 YOLO DETECTION API
# =======================================================
@app.route("/api/object_detect", methods=["POST"])
def object_detect_api():
    model = load_yolo()
    
    if model is None:
        return jsonify({"ok": False, "msg": "Model YOLO gagal load!"}), 500
    
    if "image" not in request.files:
        return jsonify({"ok": False, "msg": "Tidak ada file gambar"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()

    # Convert image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"ok": False, "msg": "Gambar tidak bisa dibaca"}), 400

    # YOLO inference
    results = model(img)

    objects = []
    found = False

    for result in results:
        for box in result.boxes:
            found = True
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            cls = int(box.cls.item())
            label = model.names.get(cls, str(cls))

            objects.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 200, 255), 3)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 200, 255),
                2
            )

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode()

    return jsonify({
        "ok": True,
        "found": found,
        "objects": objects,
        "image": img_base64
    })


# =======================================================
# LOCAL RUN
# =======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
