import cv2
citra = cv2.imread('sample.png')
cv2.imshow('hmbtn', citra)

# array 3 dimensi (631, 617, 3) = N x M x b
print(citra.shape)

# blue 
blue_layer = citra[:,:,0]
print(blue_layer)
# array 2 dimensi (631, 617)  untuk layer biru saja
print(blue_layer.shape)

# green 
green_layer = citra[:,:,1]
print(green_layer)

# red 
red_layer = citra[:,:,2]
print(red_layer)

cv2.waitKey()