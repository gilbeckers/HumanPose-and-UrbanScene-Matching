import cv2
from matplotlib import pyplot as plt
import feature_operations

img_model = cv2.imread('../img/pisa9.jpg', 0)
img_input = cv2.imread('../img/pisa_chapel.jpg',0)


sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_input,None) # returns keypoints and descriptors
# draw only keypoints location,not size and orientation
img3 = cv2.drawKeypoints(img_input, kp1, None, color=(0,255,0), flags=0)

kp2, des2 = sift.detectAndCompute(img_model,None) # returns keypoints and descriptors
# draw only keypoints location,not size and orientation
img4 = cv2.drawKeypoints(img_model, kp2, None, color=(0,255,0), flags=0)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
ax1.imshow(img4)
# ax1.set_title(model_image_name + ' (model)')
ax1.set_title("model")
ax1.axis("off")

ax2.axis("off")
ax2.imshow(img3)
ax2.set_title("input")
plt.show()

(matchesMask, input_image_homo, good, model_pts, input_pts, perspective_trans_matrix) = feature_operations.flann_matching(des2, des1, kp2, kp1, img_model, img_input)

# ---------------- DRAW MATCHES  -------------------------------
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
plt.figure()
img3 = cv2.drawMatches(img_model,kp2,input_image_homo,kp1,good,None,**draw_params)
#img3 = cv2.drawMatches(model_image,kp_model,cv2.imread('img/' + input_name + '.' + img_tag ,0),kp_input,good,None,**draw_params)
plt.imshow(img3) # Draw greyvalue images
plt.show()
