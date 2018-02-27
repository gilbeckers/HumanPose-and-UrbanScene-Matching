import cv2
from matplotlib import pyplot as plt

import numpy as np

img = cv2.imread('pisa9.jpg', 0)
img2 = cv2.imread('pisa101.jpg', 0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
kp2, des2 = surf.detectAndCompute(img2, None)

print(len(kp))

print( surf.getHessianThreshold() )


print( len(kp) )

img1_res = cv2.drawKeypoints(img,kp,None,(255,0,0),0)
img2_res = cv2.drawKeypoints(img2,kp2,None,(255,0,0),0)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
ax1.imshow(img1_res)
# ax1.set_title(model_image_name + ' (model)')
ax1.set_title("model")

ax2.imshow(img2_res)
ax2.set_title("input")

plt.show()

#cv2.imshow(img2)
#cv2.show()




