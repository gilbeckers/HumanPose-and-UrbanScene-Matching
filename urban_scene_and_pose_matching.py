# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import clustering
import affine_transformation
import matplotlib.patches as mpatches
import util
import feature_operations

MIN_MATCH_COUNT = 10

model_name = 'trap7'   # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'trap1'  # goeie : "pisa10"  taj4  # trap2     trap3
img_tag = 'jpg'

# goeie voorbeelden zijn pisa9 en pisa10
model_image = cv2.imread('img/' + model_name + '.' + img_tag ,0)
input_image = cv2.imread('img/' + input_name + '.' + img_tag ,0)
# Resize
# TODO: optimize + altijd noodzakelijk om te resizen?
model_image, input_image = util.resize_img(model_image, input_image)

list_poses = {
    # 1: linkerpols   2:rechterpols
    "blad1": np.array([[113, 290], [179, 290]]),
    "pisa_chapel": np.array([[113, 290], [179, 290]]),
    "blad2": np.array([[113, 290], [179, 290]]),
    "trap1": np.array([[113, 290], [179, 290]]),  #np.array([[113, 290], [179, 290]], np.float32)  # trap1
    "trap2": np.array([[127, 237], [206, 234]]),  # np.array([[127, 237], [206, 234]], np.float32)  # trap1
    "trap3": np.array([[218, 299], [280, 300]]),
    "trap4": np.array([[254, 248], [293, 253]]),
    "trap7": np.array([[254, 248], [293, 253]]),
    "trap8": np.array([[150, 230],[225, 225]]),  #trap8   rpols, renkel, lenkel
    "trap9": np.array([[136, 230], [217, 221]]),  #trap9  rpols, renkel, lenkel  , [343, 542]
    "taj3": np.array([[391, 92], [429, 126]]),  # taj3  enkel recher pols + r elbow     #np.array([[391, 92], [517, 148]])  # taj3  enkel recher pols + nek
    "taj4": np.array([[303, 37], [347, 70]]),   # taj4 enkel rechter pols + r elbow     #np.array([[303, 37], [412, 90]])  # taj4 enkel rechter pols + nek
    "pisa9": np.array([[152, 334], [153, 425]]),  #  np.array([[256, 362], [247, 400]], np.float32)
    "pisa10" : np.array([[256, 362], [247, 400]]),
    "pisa101" : np.array([[256, 362], [247, 400]])
    }

model_pose_features = list_poses[model_name]
input_pose_features = list_poses[input_name]

assert model_pose_features.shape == input_pose_features.shape

# --------- SIFT FEATURE DETETCION & DESCRIPTION ------------------------
kp_model, des_model = feature_operations.sift_detect_and_compute(model_image)
kp_input, des_input = feature_operations.sift_detect_and_compute(input_image)

# --------- FEATURE MATCHING : FLANN MATCHER -------------------
#(matchesMask, model_image_homo, good, model_pts, input_pts) = feature_operations.flann_matching(des_model, des_input, kp_model, kp_input, model_image, input_image)
(matchesMask, input_image_homo, good, model_pts, input_pts, perspective_trans_matrix) = feature_operations.flann_matching(des_model, des_input, kp_model, kp_input, model_image, input_image)

# ---------------- DRAW MATCHES  -------------------------------
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
plt.figure()
img3 = cv2.drawMatches(model_image,kp_model,input_image_homo,kp_input,good,None,**draw_params)
#img3 = cv2.drawMatches(model_image,kp_model,cv2.imread('img/' + input_name + '.' + img_tag ,0),kp_input,good,None,**draw_params)
plt.imshow(img3) # Draw greyvalue images
plt.show(block=False)


# ------- APPLY MASK AND ONLY KEEP THE FEATURES OF THE FIRST FOUND HOMOGRAPHY -----
# Convert mask array to an array of boolean type [ False True True False ... ]
my_mask = np.asarray(matchesMask).astype(bool)
# Apply mask to feature points of destination img, so only the features of homography remain
my_model_pts = model_pts[np.array(my_mask)]
my_input_pts = input_pts[np.array(my_mask)]

# Reshape to simple 2D array [ [x,y] , [x',y'], ... ]
model_pts_2D = np.squeeze(my_model_pts[:])
input_pts_2D = np.squeeze(my_input_pts[:])

print("Total good matches: ", len(good))
print("Total matches for bouding box des: ", len(my_model_pts))
print("Total matches for bouding box source: ", len(my_input_pts))

# ------------------   Validate Homography / Perspective matrix  -------------------------------------

#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
my_model_pts2 = np.float32(model_pts_2D).reshape(-1,1,2)  # bit reshapeing so the cv2.perspectiveTransform() works
model_transform_pts_2D = cv2.perspectiveTransform(my_model_pts2,perspective_trans_matrix) #transform(input_pts_2D)
model_transform_pts_2D = np.squeeze(model_transform_pts_2D[:]) # strip 1 dimension

## 1. Check the Reprojection error:  https://stackoverflow.com/questions/11053099/how-can-you-tell-if-a-homography-matrix-is-acceptable-or-not
# https://en.wikipedia.org/wiki/Reprojection_error
# Calc the euclidean distance for the first 3 features
reprojection_error = ( ((model_transform_pts_2D[0, 0] - input_pts_2D[0,0]) ** 2 + (model_transform_pts_2D[0, 1] - input_pts_2D[0,1]) ** 2)
      + ((model_transform_pts_2D[1, 0] - input_pts_2D[1,0]) ** 2 + (model_transform_pts_2D[1, 1] - input_pts_2D[1,1]) ** 2)
      + ((model_transform_pts_2D[2, 0] - input_pts_2D[2,0]) ** 2 + (model_transform_pts_2D[2, 1] - input_pts_2D[2,1]) ** 2)) ** 0.5

print("--- reprojection distance: " , reprojection_error)

## 2. Determinant of 2x2 matrix  (source: https://dsp.stackexchange.com/questions/17846/template-matching-or-object-recognition)
# Property of an affine (and projective?) transformation:  (source: http://answers.opencv.org/question/2588/check-if-homography-is-good/)
#   -If the determinant of the top-left 2x2 matrix is > 0 the transformation is orientation-preserving.
#   -Else if the determinant is < 0, it is orientation-reversing.
det = perspective_trans_matrix[0,0] * perspective_trans_matrix[1,1] - perspective_trans_matrix[0,1] * perspective_trans_matrix[1,0]
print("----- determinant of topleft 2x2 matrix: " , det)
if det<0:
    print("determinant<0, homography unvalid")
    exit()


'''
# understand affine transfromation: https://stackoverflow.com/questions/10667834/trying-to-understand-the-affine-transform/


# How to check if obtained homography matrix is good?  https://stackoverflow.com/questions/14954220/how-to-check-if-obtained-homography-matrix-is-good
1. Homography should preserve the direction of polygonal points. 
Design a simple test. points (0,0), (imwidth,0), (width,height), (0,height) represent a 
quadrilateral with clockwise arranged points. Apply homography on those points and see if 
they are still clockwise arranged if they become counter clockwise your homography is flipping (mirroring) 
the image which is sometimes still ok. But if your points are out of order than you have a "bad homography"

2. The homography doesn't change the scale of the object too much. For example if you expect it to shrink or 
enlarge the image by a factor of up to X, just check this rule. 
Transform the 4 points (0,0), (imwidth,0), (width-1,height), (0,height) with homography and 
calculate the area of the quadrilateral (opencv method of calculating area of polygon) if 
the ratio of areas is too big (or too small), you probably have an error.

3. Good homography is usually uses low values of perspectivity. Typically if the size of 
the image is ~1000x1000 pixels those values should be ~0.005-0.001. High perspectivity 
will cause enormous distortions which are probably an error. If you don't know where those values 
are located read my post: trying to understand the Affine Transform . 
It explains the affine transform math and the other 2 values are perspective parameters.
'''

## 3. copied from https://dsp.stackexchange.com/questions/17846/template-matching-or-object-recognition
#other explanation: https://dsp.stackexchange.com/questions/1990/filtering-ransac-estimated-homographies
N1 = (perspective_trans_matrix[0,0] * perspective_trans_matrix[0,0] + perspective_trans_matrix[0,1] * perspective_trans_matrix[0,1]) ** 0.5
if N1 > 4 or N1 < 0.1:
    print("not ok 1")
    #exit()
N2 = (perspective_trans_matrix[1,0] * perspective_trans_matrix[1,0] + perspective_trans_matrix[1,1] * perspective_trans_matrix[1,1]) ** 0.5
if N2 > 4 or N2 < 0.1:
    print("not ok 2")
    #exit()
N3 = (perspective_trans_matrix[2,0] * perspective_trans_matrix[2,0] + perspective_trans_matrix[2,1] * perspective_trans_matrix[2,1]) ** 0.5
if N3 > 0.002:
    print("not ok 3")
    #exit()

markersize = 3

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
implot = ax1.imshow(model_image)
# ax1.set_title(model_image_name + ' (model)')
ax1.set_title("model")
ax1.plot(*zip(*model_pts_2D), marker='o', color='magenta', ls='', label='model',
         ms=markersize)  # ms = markersize
red_patch = mpatches.Patch(color='magenta', label='model')
ax1.legend(handles=[red_patch])

# ax2.set_title(input_image_name + ' (input)')
ax2.set_title("input")
ax2.imshow(input_image)
ax2.plot(*zip(*input_pts_2D), marker='o', color='r', ls='', ms=markersize)
ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

ax3.set_title("transformation of model onto input err: " + str(round(reprojection_error,3)))
ax3.imshow(input_image)
ax3.plot(*zip(*input_pts_2D), marker='o', color='magenta', ls='', label='model',
         ms=markersize)  # ms = markersize
ax3.plot(*zip(*model_transform_pts_2D), marker='o', color='b', ls='', ms=markersize)
ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
                    mpatches.Patch(color='magenta', label='model')])
# plt.tight_layout()
plt.show(block=False)





# ------------- Kmeans - CLUSTERING THE FEATURES -------------
# --> If more than one building is detected ie pisa9.jpg & pisa10.jpg => cluster & seperate in individual buildings
# define criteria and apply kmeans()
clustered_features, one_building = clustering.kmean(model_pts_2D, input_pts_2D)

# Reduce dimensions
clustered_model_features = np.squeeze(clustered_features[0])  # first elements in array are model features
clustered_input_features = np.squeeze(clustered_features[1])  # second elements in array are model features

feature_operations.plot_features(clustered_model_features, clustered_input_features, one_building, model_image, input_image)

'''
feature_operations.affine_transform_urban_scene_and_pose(one_building, model_pose_features, input_pose_features,
                                                         clustered_input_features, clustered_model_features,model_image, input_image, perspective_trans_matrix)
'''

plt.show()










