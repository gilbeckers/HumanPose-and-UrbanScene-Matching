# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from old_scripts import affine_transformation

MIN_MATCH_COUNT = 10
# goeie voorbeelden zijn pisa9 en pisa10
img1 = cv2.imread('img/pisa9.jpg',0)          # queryImage
img2 = cv2.imread('img/pisa10.jpg',0) # trainImage

p9_r_pols = np.array([[152, 334]])
p9_l_pols = np.array([[153,425]])
p10_r_pols = np.array([[256, 362]])
p10_l_pols = np.array([[247, 400]])

print("pols = ", p9_l_pols)


# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# # Initiate ORB detector
#orb = cv2.ORB_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
#
# find the keypoints with ORB
# kp1 = orb.detect(img1,None)
# # compute the descriptors with ORB
# kp1, des1 = orb.compute(img1, kp1)
# # find the keypoints with ORB
# kp2 = orb.detect(img2,None)
# # compute the descriptors with ORB
# kp2, des2 = orb.compute(img2, kp2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)  #tresh : 5
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


# Convert mask array to an array of boolean type [ False True True False ... ]
my_mask = np.asarray(matchesMask).astype(bool)
# Apply mask to feature points of destination img, so only the features of homography remain
my_dst_pts = dst_pts[np.array(my_mask)]
my_scr_pts = src_pts[np.array(my_mask)]

# Reshape to simple 2D array [ [x,y] , [x',y'], ... ]
dst_pts_2D = np.squeeze(my_dst_pts[:])
src_pts_2D = np.squeeze(my_scr_pts[:])


print("TOtal good matches: " , len(good))
print("Total matches for bouding box des: ", len(my_dst_pts))
print("Total matches for bouding box source: ", len(my_scr_pts))

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(dst_pts_2D,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
ret2,label2,center2=cv2.kmeans(src_pts_2D,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print("center : ", center)
distance_centers =  np.sqrt((center[0][0]-center[1][0])**2 + (center[0][1]-center[1][1])**2)
print("distance ceners: ", distance_centers)

distance_centers2 =  np.sqrt((center2[0][0]-center2[1][0])**2 + (center2[0][1]-center2[1][1])**2)

if(distance_centers <= 150): # als centers te dicht bij elkaar liggen => opniew knn met k=1
    print("---Centers te dicht bij elkaar!")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(dst_pts_2D, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = dst_pts_2D[label.ravel()==0]
B = dst_pts_2D[label.ravel()==1]

A2 = src_pts_2D[label2.ravel()==0]
B2 = src_pts_2D[label2.ravel()==1]


# Calc nearest point to pols-punten in matching features

index_closest_points = 33  # bij pisa9 & pisa10 is da 33
distance_pols_links = ( (A[:, 0] - p10_l_pols[0][0])** 2 + (A[:, 1] - p10_l_pols[0][1]) ** 2) ** 0.5
min_distance_pols_links = np.argmin(distance_pols_links)
distance_pols_input_links = ( (A2[index_closest_points, 0] - p9_l_pols[0][0])** 2 + (A2[index_closest_points, 1] - p9_l_pols[0][1]) ** 2) ** 0.5

distance_pols_rechts = ( (A[:, 0] - p10_r_pols[0][0])** 2 + (A[:, 1] - p10_r_pols[0][1]) ** 2) ** 0.5
min_distance_pols_rechts = np.argmin(distance_pols_rechts)
distance_pols_input_rechts = ( (A2[min_distance_pols_rechts, 0] - p9_r_pols[0][0])** 2 + (A2[min_distance_pols_rechts, 1] - p9_r_pols[0][1]) ** 2) ** 0.5

print("index: " , min_distance_pols_rechts)

print("distance model links: ", distance_pols_links[min_distance_pols_links])
print("distance input links: ", distance_pols_input_links)

print("distance model rechts: ", distance_pols_rechts[min_distance_pols_rechts])
print("distance input rechts: ", distance_pols_input_rechts)

# Plot the data

plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(p10_l_pols[:,0], p10_l_pols[:,1], c='magenta')
plt.scatter(A[min_distance_pols_links,0], A[min_distance_pols_links,1], c='navy')
plt.scatter(p10_r_pols[:,0], p10_r_pols[:,1], c='magenta')
plt.scatter(A[min_distance_pols_rechts,0], A[min_distance_pols_rechts,1], c='navy')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')



'''--BOUNDING BOX added--'''
rect = cv2.minAreaRect(A)
box = cv2.boxPoints(rect)

# Bounding box recht trekken => zijden staan loodrecht op assen
# TODO zou in feite parallel met horizon moeten zijn? Voor bv als fout schuin wordt getrokken? ??
# ---> en wat met scheve gebouwen?

xmax, ymax = A.max(axis=0)
xmin, ymin = A.min(axis=0)
box[0] = [xmin, ymax]
box[1] = [xmin, ymin]
box[2] = [xmax, ymin]
box[3] = [xmax, ymax]

box = np.int0(box)
'''--END BOUNDING BOX'''
img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

'''--BOUNDING BOX added--'''
rect = cv2.minAreaRect(B)
box = cv2.boxPoints(rect)
box = np.int0(box)
'''--END BOUNDING BOX'''
img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

plt.xlabel('Height'),plt.ylabel('Weight')
plt.imshow(img2)
#plt.show()
plt.show(block=False)
plt.figure()

# -------------Plot the src data
plt.scatter(A2[:,0],A2[:,1])
plt.scatter(B2[:,0],B2[:,1],c = 'r')
plt.scatter(p9_l_pols[:,0], p9_l_pols[:,1], c='magenta')
plt.scatter(A2[min_distance_pols_links,0], A2[min_distance_pols_links,1], c='navy')
plt.scatter(p9_r_pols[:,0], p9_r_pols[:,1], c='magenta')
plt.scatter(A2[min_distance_pols_rechts,0], A2[min_distance_pols_rechts,1], c='navy')
plt.scatter(center2[:,0],center2[:,1],s = 80,c = 'y', marker = 's')


#  int cv::rotatedRectangleIntersection
# https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9

'''--BOUNDING BOX added--'''
rect2 = cv2.minAreaRect(A2)
box2 = cv2.boxPoints(rect2)

# Bounding box recht trekken => zijden staan loodrecht op assen
# TODO zou in feite parallel met horizon moeten zijn? Voor bv als fout schuin wordt getrokken? ??
# ---> en wat met scheve gebouwen?

xmax2, ymax2 = A2.max(axis=0)
xmin2, ymin2 = A2.min(axis=0)
box2[0] = [xmin2, ymax2]
box2[1] = [xmin2, ymin2]
box2[2] = [xmax2, ymin2]
box2[3] = [xmax2, ymax2]

box2 = np.int0(box2)
'''--END BOUNDING BOX'''
img1 = cv2.drawContours(img1, [box2], 0, (0, 0, 255), 2)

'''--BOUNDING BOX added--'''
rect2 = cv2.minAreaRect(B2)
box2 = cv2.boxPoints(rect2)
box2 = np.int0(box2)
'''--END BOUNDING BOX'''
img1 = cv2.drawContours(img1, [box2], 0, (0, 0, 255), 2)

plt.xlabel('Height'),plt.ylabel('Weight')
plt.imshow(img1)
#plt.show()
plt.show(block=False)
plt.figure()


plt.imshow(img3, 'gray')
plt.show(block=False)


#### CALC AFFINE TRANSFORMATION ####
# Calc affine trans between the wrest points and some random feature points of the building
# The question is: WHICH feature points should we take??
# An option is to go for the "best matches" (found during featuring-matching)
# An other option is just to take an certain number of random matches

# Third option would be to take all the building feature points,
# but that would probably limit transformation in aspect of the mutual spatial
# relation between the person and the building
# TODO: other options??

#Create feature array for first person
# feature points of pisa tower are in A
# feautes van pols =
# p9_r_pols = np.array([[152, 334]])
# p9_l_pols = np.array([[153,425]])

input_features = np.array([[152,334], [153,425]])
#input_features = np.append(input_features, A2, 0)

input_features = np.append(input_features, [A2[0]], 0 )
input_features = np.append(input_features, [A2[2]], 0 )
input_features = np.append(input_features, [A2[6]], 0 )
input_features = np.append(input_features, [A2[14]], 0 )




#p10_r_pols = np.array([[256, 362]])
#p10_l_pols = np.array([[247, 400]])
output_features = np.array([[256,362], [247, 400]])
#output_features = np.append(output_features, A, 0)

output_features = np.append(output_features, [A[0]], 0 )
output_features = np.append(output_features, [A[2]], 0 )
output_features = np.append(output_features, [A[6]], 0 )
output_features = np.append(output_features, [A[14]], 0 )

print("input: " , input_features)
print("output: ", output_features)

(input_transformed, transformation_matrix) = affine_transformation.find_transformation(output_features, input_features)

markersize=3

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
implot = ax1.imshow(cv2.imread('img/pisa10.jpg',0))
#ax1.set_title(model_image_name + ' (model)')
ax1.set_title("model")
ax1.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
red_patch = mpatches.Patch(color='magenta', label='model')
ax1.legend(handles=[red_patch])

#ax2.set_title(input_image_name + ' (input)')
ax2.set_title("input")
ax2.imshow(cv2.imread('img/pisa9.jpg',0))
ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

ax3.set_title("transformation")
ax3.imshow(cv2.imread('img/pisa10.jpg',0))
ax3.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'), mpatches.Patch(color='magenta', label='model')])

plt.show()






