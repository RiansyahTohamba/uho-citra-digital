import joblib

# Load model dan label encoder
model = joblib.load('model_penyakit_daun.pkl')
label_encoder = joblib.load('label_encoder.pkl')
print("Model dan label encoder berhasil dimuat.")

print('upload gambar')
button()
img = hasil_upload()

result = model.predict(img)
resukt