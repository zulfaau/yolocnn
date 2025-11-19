from flask import Flask, render_template, request, jsonify
import numpy as np
import re
from io import BytesIO
import base64
import os
import random
import pandas as pd

# FIX BACKEND MATPLOTLIB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# YOLO
try:
    from ultralytics import YOLO
    import cv2
    from PIL import Image
    YOLO_AVAILABLE = True
except Exception as e:
    print("YOLO ERROR:", e)
    YOLO_AVAILABLE = False

app = Flask(__name__, template_folder="templates")


# =========================
# ROUTES - PAGES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/kalkulator")
def kalkulator_page():
    return render_template("kalkulator.html")


@app.route("/generate")
def generate_page():
    return render_template("generate.html")


@app.route("/saham")
def saham_page():
    return render_template("saham.html")


@app.route("/object")
def object_page():
    return render_template("object_detection.html")


# =========================
# API: KALKULATOR LOGIC GATE
# =========================
@app.route("/api/logic", methods=["POST"])
def api_logic():
    try:
        data = request.get_json(force=True)
        a = int(data.get("a", 0))
        b = int(data.get("b", 0))
        gate = data.get("gate", "AND").upper()

        gates = {
            "AND": lambda x, y: x & y,
            "OR": lambda x, y: x | y,
            "XOR": lambda x, y: x ^ y,
            "NAND": lambda x, y: int(not (x & y)),
            "NOR": lambda x, y: int(not (x | y)),
            "XNOR": lambda x, y: int(not (x ^ y))
        }

        if gate not in gates:
            return jsonify({"ok": False, "error": "Gate tidak valid"}), 400

        result = gates[gate](a, b)

        truth_table = [
            {"A": x, "B": y, "Output": gates[gate](x, y)}
            for x in [0, 1]
            for y in [0, 1]
        ]

        return jsonify({"ok": True, "result": result, "truth_table": truth_table})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# API: SAHAM
# =========================
@app.route("/api/predict_stock", methods=["POST"])
def api_predict_stock():
    try:
        data = request.get_json(silent=True) or request.form
        kode = (data.get("kode") or data.get("symbol") or "").upper()
        days = int(data.get("days", 7))

        csv_candidate = os.path.join(os.path.dirname(__file__), "Data Historis BBCA_Test2.csv")
        if os.path.exists(csv_candidate):
            df = pd.read_csv(csv_candidate)
            for col in ["Close", "close", "Harga Penutupan", "harga"]:
                if col in df.columns:
                    price_col = col
                    break
            else:
                price_col = df.columns[0]

            prices = df[price_col].dropna().tolist()
            if len(prices) >= days:
                actual = prices[-days:]
                preds = []
                for i in range(len(actual)):
                    if i == 0:
                        preds.append(actual[i])
                    else:
                        delta = actual[i] - actual[i - 1]
                        preds.append(preds[-1] + delta * random.uniform(0.8, 1.2))

                labels = [f"Hari ke-{i+1}" for i in range(len(preds))]

                return jsonify({
                    "ok": True,
                    "symbol": kode or "BBCA",
                    "model": "Simple Forecast",
                    "days": len(preds),
                    "labels": labels,
                    "actual": actual,
                    "predictions": preds,
                    "prediksi_terakhir": f"Rp {preds[-1]:,.0f}".replace(",", ".")
                })

        last = random.uniform(1000, 10000)
        preds = [round(last * (1 + random.uniform(-0.02, 0.02) * i/10), 2) for i in range(days)]

        labels = [f"Hari ke-{i+1}" for i in range(days)]
        return jsonify({
            "ok": True,
            "symbol": kode or "DEMO",
            "model": "Random Demo",
            "days": days,
            "labels": labels,
            "actual": [],
            "predictions": preds,
            "prediksi_terakhir": f"Rp {preds[-1]:,.0f}"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# API: YOLO OBJECT DETECTION
# =========================
@app.route("/api/object_detect", methods=["POST"])
def api_object_detect():
    try:
        if not YOLO_AVAILABLE:
            return jsonify({"ok": False, "msg": "YOLOv8 tidak tersedia di environment ini"}), 500

        yolo_model = YOLO("models/yolov8n.pt")

        if "image" not in request.files:
            return jsonify({"ok": False, "msg": "Tidak ada gambar"}), 400

        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        results = yolo_model(np.array(img), verbose=False)[0]

        np_img = np.array(img)
        detected_objects = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(np_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(np_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            detected_objects.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

        pil_out = Image.fromarray(np_img)
        buff = BytesIO()
        pil_out.save(buff, format="JPEG")
        base64_img = base64.b64encode(buff.getvalue()).decode()

        return jsonify({
            "ok": True,
            "found": len(detected_objects) > 0,
            "image": base64_img,
            "objects": detected_objects
        })

    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("Starting app...")
    app.run(host="0.0.0.0", port=port)
