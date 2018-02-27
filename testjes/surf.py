import cv2
import matplotlib as plt

import numpy as np

img = cv2.imread('pisa9.jpg')

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

print(len(kp))

print( surf.getHessianThreshold() )

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
#surf.setHessianThreshold(50000)
# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
cv2.imwrite('key.jpg',img2)

cv2.imshow(img2)
cv2.show()




