import joblib
import os
import cv2
import numpy as np
from feature_utils import extract_features

# --- Load model dan label encoder ---
model = joblib.load('model_penyakit_daun.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# --- Direktori gambar untuk diuji ---
test_dir = "data-test"
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("Tidak ada gambar yang ditemukan di folder 'data-test'.")
    exit()

print("=== Hasil Prediksi ===")
for fname in image_files:
    fpath = os.path.join(test_dir, fname)
    try:
        features = extract_features(fpath).reshape(1, -1)
        prediction = model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]
        print(f"{fname}: {label}")
    except Exception as e:
        print(f"{fname}: Gagal diproses ({e})")
