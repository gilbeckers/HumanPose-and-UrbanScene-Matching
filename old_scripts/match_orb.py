# further information:
# * http://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python
# * http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html#feature-homography
# * http://stackoverflow.com/questions/9539473/opencv-orb-not-finding-matches-once-rotation-scale-invariances-are-introduced
# * OpenCV 2 Computer Vision Application Programming Cookbook, Chapter 9
import cv2
import scipy as sp
import numpy as np


ratio = 0.65

""" Clear matches for which NN ratio is > than threshold """
def filter_distance(matches):
    dist = [m.distance for m in matches]
    thres_dist = (sum(dist) / len(dist)) * ratio

    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < thres_dist]
    print('#selected matches:%d (out of %d)' % (len(sel_matches), len(matches)))
    return sel_matches

""" keep only symmetric matches """
def filter_asymmetric(matches, matches2):
    sel_matches = []
    for match1 in matches:
        for match2 in matches2:
            if k_ftr[match1.queryIdx] == k_ftr[match2.trainIdx] and k_scene[match1.trainIdx] == k_scene[match2.queryIdx]:
                sel_matches.append(match1)
                break
    return sel_matches

# Todo: filter_ransac

def filter_matches(matches, matches2):
    matches = filter_distance(matches)
    matches2 = filter_distance(matches2)
    return filter_asymmetric(matches, matches2)




img1_path = "img/pisa9.jpg"
img2_path = "img/pisa10.jpg"

img_scene = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img_ftr = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create() #cv2.FeatureDetector_create("ORB") #SURF


#descriptor = cv2.DescriptorExtractor_create("ORB") #BRIEF
matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming") #FlannBased #BruteForce-Hamming


# detect keypoints
kp_scene = detector.detect(img_scene)
kp_ftr = detector.detect(img_ftr)

print('#keypoints in image1: %d, image2: %d' % (len(kp_scene), len(kp_ftr)))

# descriptors
k_scene, d_scene = detector.compute(img_scene, kp_scene)
k_ftr, d_ftr = detector.compute(img_ftr, kp_ftr)



print('#keypoints in image1: %d, image2: %d' % (len(d_scene), len(d_ftr)))

# match the keypoints
matches = matcher.match(d_scene, d_ftr)
matches2 = matcher.match(d_ftr, d_scene)

# visualize the matches
print('#matches:', len(matches))
dist = [m.distance for m in matches]

print('distance: min: %.3f' % min(dist))
print('distance: mean: %.3f' % (sum(dist) / len(dist)))
print('distance: max: %.3f' % max(dist))



""" filter matches """

sel_matches = filter_matches(matches,matches2)


""" localize object """

h_scene, w_scene = img_scene.shape[:2]
h_ftr, w_ftr = img_ftr.shape[:2]

ftr =[]
scene = []

for m in sel_matches:
    scene.append(k_scene[m.queryIdx].pt)
    ftr.append(k_ftr[m.trainIdx].pt)

ftr = np.float32(ftr)
scene = np.float32(scene)

homography, mask = cv2.findHomography(ftr, scene, cv2.RANSAC)
ftr_corners = np.float32([[0, 0], [w_ftr, 0], [w_ftr, h_ftr], [0, h_ftr]]).reshape(1, -1, 2)
corners = np.int32( cv2.perspectiveTransform(ftr_corners, homography).reshape(-1, 2) )





""" visualization """

view = sp.zeros((max(h_scene, h_ftr), w_scene + w_ftr, 3), np.uint8)
view[:h_scene, :w_scene, 0] = img_scene
view[:h_ftr, w_scene:, 0] = img_ftr
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]



for m in sel_matches:
    # draw the keypoints
    color = tuple([sp.random.randint(0, 255) for _ in range(3)])
    cv2.line(view, (int(k_scene[m.queryIdx].pt[0]), int(k_scene[m.queryIdx].pt[1])),
        (int(k_ftr[m.trainIdx].pt[0] + w_scene), int(k_ftr[m.trainIdx].pt[1])), color, 2)



cv2.polylines(view, [np.int32([c+[w_scene,0] for c in ftr_corners])], True, (0, 255, 0), 2)
cv2.polylines(view, [corners], True, (0, 255, 0), 2)

cv2.imshow("view", view)
cv2.imwrite("output.jpg", view)
cv2.waitKey()