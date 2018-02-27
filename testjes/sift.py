import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pisa9.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None) # returns keypoints and descriptors
# draw only keypoints location,not size and orientation
img3 = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)

plt.imshow(img3), plt.show()