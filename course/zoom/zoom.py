import cv2

# Membaca gambar resolusi tinggi dan rendah
high_res = cv2.imread('low-res-257-144.jpg')   
low_res = cv2.imread('high-res-1152-648.jpg') 

# Memperbesar gambar (zoom in) menggunakan resize
scale_factor = 4
high_res_zoom = cv2.resize(high_res, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
low_res_zoom = cv2.resize(low_res, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Menampilkan hasil
cv2.imshow('High Resolution Zoom', high_res_zoom)
cv2.imshow('Low Resolution Zoom', low_res_zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()
