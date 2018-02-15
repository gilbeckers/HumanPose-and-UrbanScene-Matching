# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import clustering
import affine_transformation
import matplotlib.patches as mpatches

MIN_MATCH_COUNT = 10

model_name = 'trap1'   # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'trap2'  # goeie : "pisa10"  taj4  # trap2     trap3
img_tag = 'jpg'

# 1: rechterpols  2: linkerpols
p9_pose = np.array([[152, 334], [153,425]])
p10_pose = np.array([[256, 362], [247, 400]])

# goeie voorbeelden zijn pisa9 en pisa10
model_image = cv2.imread('img/' + model_name + '.' + img_tag ,0)
input_image = cv2.imread('img/' + input_name + '.' + img_tag ,0)


## ---------------- RESIZE ------------------
# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
r = 500.0 / model_image.shape[1]
dim = (500, int(input_image.shape[0] * r))

# perform the actual resizing of the image and show it
model_image = cv2.resize(model_image, dim, interpolation = cv2.INTER_AREA)
input_image = cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)
## ---------------- END RESIZE ------------------


## -- -----  BACKGROUND DETECTION  --------------------

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(model_image,kernel,iterations = 1, anchor=(100, 300))

cv2.imshow('frame',erosion)

## -- -----  BACKGROUND DETECTION  --------------------

model_pose = p9_pose
input_pose = p10_pose


# --------- SIFT FEATURE DETETCION & DESCRIPTION ------------------------
# Initiate SIFT detector  # TODO: what is best? ORB SIFT  &&& FLANN?
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp_model, des_model = sift.detectAndCompute(model_image,None)
kp_input, des_input = sift.detectAndCompute(input_image,None)

# --------- FEATURE MATCHING : FLANN MATCHER -------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_model,des_input,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    model_pts = np.float32([ kp_model[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    input_pts = np.float32([ kp_input[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Find only the good, corresponding points (lines of matching points may not cross)
    M, mask = cv2.findHomography(model_pts, input_pts, cv2.RANSAC,5.0)  #tresh : 5
    matchesMask = mask.ravel().tolist()
    h,w = model_image.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    input_image_homo = cv2.polylines(input_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)  # draw homography square
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(model_image,kp_model,input_image_homo,kp_input,good,None,**draw_params)
plt.imshow(img3) # Draw greyvalue images
plt.show(block=False)
plt.figure()

# ------- APPLY MASK AND ONLY KEEP THE FEATURES OF THE FIRST FOUND HOMOGRAPHY -----
# Convert mask array to an array of boolean type [ False True True False ... ]
my_mask = np.asarray(matchesMask).astype(bool)
# Apply mask to feature points of destination img, so only the features of homography remain
my_model_pts = model_pts[np.array(my_mask)]
my_input_pts = input_pts[np.array(my_mask)]

# Reshape to simple 2D array [ [x,y] , [x',y'], ... ]
model_pts_2D = np.squeeze(my_model_pts[:])
input_pts_2D = np.squeeze(my_input_pts[:])

print("TOtal good matches: " , len(good))
print("Total matches for bouding box des: ", len(my_model_pts))
print("Total matches for bouding box source: ", len(my_input_pts))

# ------------- Kmeans - CLUSTERING THE FEATURES -------------
# --> If more than one building is detected ie pisa9.jpg & pisa10.jpg => cluster & seperate in individual buildings
# define criteria and apply kmeans()
clustered_features, one_building = clustering.kmean(model_pts_2D, input_pts_2D)

clustered_model_features = clustered_features[0]
clustered_input_features = clustered_features[1]



if one_building: # Take the first found homography, no more computations needed
    #Reduce dimensions
    clustered_model_features = np.squeeze(clustered_features[0])
    clustered_input_features = np.squeeze(clustered_features[1])

    print("one building only")
    plt.scatter(clustered_model_features[:,0],clustered_model_features[:,1])
    #plt.scatter(model_center[:,0],model_center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.imshow(model_image)
    plt.show(block=False)
    plt.figure()

    plt.scatter(clustered_input_features[:,0],clustered_input_features[:,1])
    #plt.scatter(input_center[:,0],input_center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.imshow(cv2.imread('img/' + input_name + '.' + img_tag))
    plt.show(block=False)
    plt.figure()

    #-------------  CALC AFFINE TRANSFORMATION  ------------------##
    # Calc affine trans between the wrest points and some random feature points of the building
    # The question is: WHICH feature points should we take??
    # An option is to go for the "best matches" (found during featuring-matching)
    # An other option is just to take an certain number of random matches

    # Third option would be to take all the building feature points,
    # but that would probably limit transformation in aspect of the mutual spatial
    # relation between the person and the building
    # TODO: other options??

    # Create feature array for first person
    # feature points of pisa tower are in A
    # feautes van pols =

    # p9_r_pols = np.array([[152, 334]])  #pisa9
    # p9_l_pols = np.array([[153,425]])
    # p10_r_pols = np.array([[256, 362]])   #pisa10
    # p10_l_pols = np.array([[247, 400]])

    input_features = np.array([[152, 334], [153, 425]])  #pisa9
    output_features = np.array([[256, 362], [247, 400]]) #pisa10



    input_features =  np.array([[391,92]])  #taj3  enkel recher pols
    input_features = np.array([[463, 89]]) # foute locatie
    input_features = np.array([[391, 92], [517, 148]])  # taj3  enkel recher pols + nek
    input_features = np.array([[391, 92], [429, 126]])  # taj3  enkel recher pols + r elbow

    output_features = np.array([[303,37]]) #taj4 enkel rechter pols
    output_features = np.array([[303, 37],[412, 90]])  # taj4 enkel rechter pols + nek
    output_features = np.array([[303, 37], [347, 70]])  # taj4 enkel rechter pols + r elbow



    input_features = np.append(input_features, [clustered_input_features[0]], 0)
    input_features = np.append(input_features, [clustered_input_features[2]], 0)
    input_features = np.append(input_features, [clustered_input_features[6]], 0)
    input_features = np.append(input_features, [clustered_input_features[14]], 0)

    output_features = np.append(output_features, [clustered_model_features[0]], 0)
    output_features = np.append(output_features, [clustered_model_features[2]], 0)
    output_features = np.append(output_features, [clustered_model_features[6]], 0)
    output_features = np.append(output_features, [clustered_model_features[14]], 0)

    (input_transformed, transformation_matrix) = affine_transformation.find_transformation(output_features,
                                                                                           input_features)

    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("transformation")
    ax3.imshow(model_image)
    ax3.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
                        mpatches.Patch(color='magenta', label='model')])
    #plt.tight_layout()
    plt.show(block=False)

    plt.figure()





else: # More than one building
    for feat in clustered_model_features:
        plt.scatter(feat[:, 0], feat[:, 1])
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.imshow(model_image)
    plt.show(block=False)
    plt.figure()


    for feat in clustered_input_features:
        plt.scatter(feat[:, 0], feat[:, 1])
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.imshow(input_image)
    plt.show(block=False)
    plt.figure()

    # -------------  CALC AFFINE TRANSFORMATION  ------------------##

    # p9_r_pols = np.array([[152, 334]])  #pisa9
    # p9_l_pols = np.array([[153,425]])
    # p10_r_pols = np.array([[256, 362]])   #pisa10
    # p10_l_pols = np.array([[247, 400]])

    input_features = np.array([[256, 362], [247, 400]])   # pisa9
    output_features = np.array([[152, 334], [153, 425]]) # pisa10

    input_features = np.array([[127, 237], [206, 234]])  # trap1
    #input_features = np.array([[218, 299], [280, 300]])  # trap3

    output_features = np.array([[116, 289], [188, 284]])  # trap2

    object_index = 0

    input_features = np.append(input_features, [clustered_input_features[object_index][0]], 0)
    input_features = np.append(input_features, [clustered_input_features[object_index][2]], 0)
    input_features = np.append(input_features, [clustered_input_features[object_index][6]], 0)
    input_features = np.append(input_features, [clustered_input_features[object_index][14]], 0)

    output_features = np.append(output_features, [clustered_model_features[object_index][0]], 0)
    output_features = np.append(output_features, [clustered_model_features[object_index][2]], 0)
    output_features = np.append(output_features, [clustered_model_features[object_index][6]], 0)
    output_features = np.append(output_features, [clustered_model_features[object_index][14]], 0)

    (input_transformed, transformation_matrix) = affine_transformation.find_transformation(output_features,
                                                                                           input_features)

    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("transformation")
    ax3.imshow(model_image)
    ax3.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
                        mpatches.Patch(color='magenta', label='model')])
    # plt.tight_layout()
    plt.show(block=False)

    plt.figure()



plt.show()
cv2.waitKey(0)









