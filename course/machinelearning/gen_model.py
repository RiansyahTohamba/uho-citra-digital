import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Fungsi Ekstraksi Fitur ---
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Normalisasi ukuran
    filtered = cv2.GaussianBlur(img, (5, 5), 0)  # Spasial filter
    
    # Fitur sederhana: histogram intensitas
    hist = cv2.calcHist([filtered], [0], None, [64], [0, 256]).flatten()
    hist = hist / np.sum(hist)  # Normalisasi
    return hist

# --- Load dataset ---
features = []
labels = []

base_path = "dataset"
for label in os.listdir(base_path):
    class_dir = os.path.join(base_path, label)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        try:
            feat = extract_features(fpath)
            features.append(feat)
            labels.append(label)
        except:
            print(f"Failed to process {fpath}")

# Encode label
le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Simpan model dan label encoder
joblib.dump(clf, 'model_penyakit_daun.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model berhasil disimpan.")
