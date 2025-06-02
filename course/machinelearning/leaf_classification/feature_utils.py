# feature_utils.py
import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    hist_gray = cv2.calcHist([blurred], [0], None, [64], [0, 256]).flatten()
    hist_gray = hist_gray / np.sum(hist_gray)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    hist_sobel = cv2.calcHist([sobel_mag.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
    hist_sobel = hist_sobel / np.sum(hist_sobel)
    
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    hist_laplacian = cv2.calcHist([laplacian.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
    hist_laplacian = hist_laplacian / np.sum(hist_laplacian)
    
    combined = np.concatenate([hist_gray, hist_sobel, hist_laplacian])
    return combined
