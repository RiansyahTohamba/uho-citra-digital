import cv2
citra = cv2.imread('sample.png')
cv2.imshow('hmbtn bgr', citra)

# array 3 dimensi (631, 617, 3) = N x M x b
print(citra.shape)

# blue 
blue_layer = citra[:,:,0]
print(blue_layer)
# array 2 dimensi (631, 617)  untuk layer biru saja
print(blue_layer.shape)
# blue layer saja = 8 bit, sehingga hanya menampilkan gray saja
cv2.imshow("blue layer", blue_layer)

# green 
# green layer saja = 8 bit
green_layer = citra[:,:,1]
print(green_layer)
cv2.imshow("green layer", green_layer)

# red 
red_layer = citra[:,:,2]
print(red_layer)
cv2.imshow("red layer", red_layer)


cv2.waitKey()