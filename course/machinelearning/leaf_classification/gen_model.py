import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from feature_utils import extract_features

import joblib

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

# Cek jumlah kelas
if len(set(y)) < 2:
    raise ValueError("Dataset harus memiliki setidaknya dua kelas (misal: 'sehat' dan 'tidak_sehat').")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# --- Train model (Logistic Regression) ---
clf = LogisticRegression(max_iter=1000, solver='lbfgs')
clf.fit(X_train, y_train)

# --- Evaluasi ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Simpan model dan encoder ---
joblib.dump(clf, 'model_penyakit_daun.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model berhasil disimpan.")
