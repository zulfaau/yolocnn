from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Bidirectional, Input
import re
from io import BytesIO
import base64
import os
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, jsonify


# FIX BACKEND MATPLOTLIB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional components for object detection (will try to load but safe if not available)
try:
    from ultralytics import YOLO
    import cv2
    from PIL import Image
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

app = Flask(__name__, template_folder="templates")


# ========= CLEANING =========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# ========= DATA PREP =========
def prepare_data(text, seq_length=40):
    text = clean_text(text)
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    sequences = []
    next_chars = []

    for i in range(0, len(text) - seq_length):
        sequences.append(text[i:i+seq_length])
        next_chars.append(text[i+seq_length])

    X = np.zeros((len(sequences), seq_length, len(chars)))
    y = np.zeros((len(sequences), len(chars)))

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            if char in char_to_idx:
                X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1

    return X, y, chars, char_to_idx, idx_to_char


# ========= MODEL =========
def build_model(model_name, seq_length, num_chars):
    model = Sequential()
    model.add(Input(shape=(seq_length, num_chars)))

    if model_name == "LSTM":
        model.add(LSTM(128))
    elif model_name == "GRU":
        model.add(GRU(128))
    elif model_name == "Vanilla RNN":
        model.add(SimpleRNN(128))
    elif model_name == "Bidirectional RNN":
        model.add(Bidirectional(LSTM(128)))
    else:
        raise ValueError("Model tidak dikenal")

    model.add(Dense(num_chars, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


# ========= GENERATOR =========
def generate_text(model, seed, idx_to_char, char_to_idx, length=150):
    generated = seed

    for _ in range(length):
        x_pred = np.zeros((1, len(seed), len(char_to_idx)))

        for t, char in enumerate(seed):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_char = idx_to_char[np.argmax(preds)]

        generated += next_char
        seed = generated[-len(seed):]

    return generated


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
# API: KALKULATOR (logic gate)
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
# API: GENERATE (train & generate single selected model)
# - This trains the chosen model (LSTM/GRU/RNN/Bidirectional) per request.
# - Returns generated text + plot of training loss (base64 img).
# =========================
@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.get_json(force=True)
        raw_text = data.get("text", "")
        model_name = data.get("model", "LSTM")
        seed = data.get("seed", "")
        gen_length = int(data.get("length", 100))
        epochs = int(data.get("epochs", 30))

        if len(raw_text) < 50:
            return jsonify({"ok": False, "error": "Teks terlalu pendek untuk training."}), 400

        seed_clean = clean_text(seed)
        if len(seed_clean) < 10:
            return jsonify({"ok": False, "error": "Seed minimal 10 karakter."}), 400
        if len(seed_clean) < 40:
            seed_clean = seed_clean.ljust(40, " ")

        X, y, chars, char_to_idx, idx_to_char = prepare_data(raw_text, seq_length=40)

        model = build_model(model_name, 40, len(chars))

        # train
        history = model.fit(X, y, epochs=epochs, batch_size=64, verbose=0)

        # generate
        generated_text = generate_text(model, seed_clean[:40], idx_to_char, char_to_idx, gen_length)

        # plot loss
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"])
        ax.set_title(f"Training Loss ({model_name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "ok": True,
            "generated": generated_text,
            "plot": f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">'
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# API: PREDIKSI SAHAM (simple demo using provided dataset if present)
# =========================
@app.route("/api/predict_stock", methods=["POST"])
def api_predict_stock():
    try:
        # Accept JSON or form
        data = request.get_json(silent=True) or request.form
        kode = (data.get("kode") or data.get("symbol") or "").upper()
        days = int(data.get("days", 7))

        # If user uploaded dataset file in project, try to use it
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
        # Fallback: random demo
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
# API: OBJECT DETECTION (YOLO) - optional if YOLO installed
# =========================
@app.route("/api/object_detect", methods=["POST"])
def api_object_detect():
    if not YOLO_AVAILABLE:
        yolo_model = YOLO("models/yolov8n.pt")

    if "image" not in request.files:
        return jsonify({"ok": False, "msg": "Tidak ada gambar"}), 400

    try:
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
            detected_objects.append({"label": label, "confidence": round(conf*100, 2)})

        pil_out = Image.fromarray(np_img)
        buff = BytesIO()
        pil_out.save(buff, format="JPEG")
        base64_img = base64.b64encode(buff.getvalue()).decode()

        return jsonify({"ok": True, "found": len(detected_objects)>0, "image": base64_img, "objects": detected_objects})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("Starting app from:", os.path.abspath(__file__))
    app.run(host="0.0.0.0", port=port)
