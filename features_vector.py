import cv2
import numpy as np

img = cv2.imread('data_set/gato1.jpg')
#width,height = img.shape[:2]
height = np.size(img, 0)
width = np.size(img, 1)
print(width,height)

print(len(img[0]))
print(len(img))

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
