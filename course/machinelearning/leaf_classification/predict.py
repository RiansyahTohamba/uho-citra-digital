import joblib
import os
import cv2
import numpy as np

# --- Load model dan label encoder ---
model = joblib.load('model_penyakit_daun.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# --- Fungsi ekstraksi fitur (sama seperti saat training) ---
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    filtered = cv2.GaussianBlur(img, (5, 5), 0)
    hist = cv2.calcHist([filtered], [0], None, [64], [0, 256]).flatten()
    hist = hist / np.sum(hist)
    return hist

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
