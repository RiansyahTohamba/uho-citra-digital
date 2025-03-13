import numpy as np
import matplotlib.pyplot as plt

# Parameter sinyal
Fs = 500  # Frekuensi sampling (Hz)
T = 1 / Fs  # Periode sampling
L = 1000  # Panjang sinyal

t = np.arange(0, L) * T  # Vektor waktu

# Sinyal terdiri dari dua sinusoidal dengan frekuensi 50 Hz dan 120 Hz
f1, f2 = 50, 120
sinyal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Transformasi Fourier
frekuensi = np.fft.fftfreq(L, T)  # Sumbu frekuensi
spektra = np.fft.fft(sinyal)  # Transformasi Fourier

# Ambil hanya bagian positif
pos_mask = frekuensi > 0
frekuensi_pos = frekuensi[pos_mask]
amplitudo_pos = 2 * np.abs(spektra[pos_mask]) / L

# Plot sinyal dan spektrum frekuensi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t[:200], sinyal[:200])  # Tampilkan hanya 200 sampel pertama
plt.title('Sinyal dalam Domain Waktu')
plt.xlabel('Waktu (detik)')
plt.ylabel('Amplitudo')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(frekuensi_pos, amplitudo_pos)
plt.title('Spektrum Frekuensi')
plt.xlabel('Frekuensi (Hz)')
plt.ylabel('Amplitudo')
plt.grid()

plt.tight_layout()
plt.show()
