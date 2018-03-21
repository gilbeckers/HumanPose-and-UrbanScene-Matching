import heapq
import logging

import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from common import anorm
import common
#from old_scripts import normalising, affine_transformation as at, prepocessing

MIN_MATCH_COUNT = 15
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
FILTER_RATIO = 0.8 #lagere ratio geeft minder 'good' matches

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(nfeatures=100000, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                                  firstLevel=0, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, patchSize=31)
        #cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)#cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        search_params = dict(checks=50)
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv2.FlannBasedMatcher(flann_params, search_params)  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = FILTER_RATIO):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:  #lagere ratio geeft minder 'good' matches
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1]).reshape(-1,1,2)
    p2 = np.float32([kp.pt for kp in mkp2]).reshape(-1,1,2)
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def match_and_draw(win, matcher, desc_model, desc_input, kp_model, kp_input, model_img, input_img, show_win):
    #print('matching...')
    raw_matches = matcher.knnMatch(desc_model, trainDescriptors = desc_input, k = 2) #2

    # p_model and p_input are the 'first-good' matches
    p_model, p_input, kp_pairs = filter_matches(kp_model, kp_input, raw_matches)

    if len(p_model) >= MIN_MATCH_COUNT:
        # Returns the perspective transformation matrix M: transformaion from model to inputplane
        H, mask = cv2.findHomography(p_model, p_input, cv2.RANSAC, 5.0)

        # NOTE!! THIS is persp trans from inputplane to modelplane!! used for perspective elimination
        H2, mask2 = cv2.findHomography(p_input, p_model, cv2.RANSAC, 5.0)
        logging.debug('%d / %d  inliers/matched' % (np.sum(mask), len(mask)))

        h, w = model_img.shape

        # the square that's drawn on the model. Just the prerspective transformation of the model image contours
        homography_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # print("###points: ", pts)
        homography_dst = cv2.perspectiveTransform(homography_pts, H)
        input_image_homo = cv2.polylines(input_img, [np.int32(homography_dst)], True, 255, 3,
                                         cv2.LINE_AA)  # draw homography square

        # ------- APPLY MASK AND ONLY KEEP THE FEATURES OF THE FIRST FOUND HOMOGRAPHY -----
        # ===> this are the "final-good" matches
        my_mask = np.asarray(mask).astype(bool)  # Convert mask array to an array of boolean type [ False True True False ... ]

        # Apply mask to feature points of destination img, so only the features of homography remain
        good_model_pts = p_model[np.array(my_mask)]
        good_input_pts = p_input[np.array(my_mask)]

        # Reshape to simple 2D array [ [x,y] , [x',y'], ... ]
        good_model_pts = np.squeeze(good_model_pts[:])
        good_input_pts = np.squeeze(good_input_pts[:])

        logging.debug("Raw matches: %d", len(p_model))
        logging.debug("Perspective matches: %d", len(good_model_pts))

    else:
        H, mask = None, None
        logging.debug('%d matches found, not enough for homography estimation' % len(p_model))

    # Render nice window with nice view of matches
    _vis = explore_match(win + '| (raw matches: ' + str(len(p_model)) + '  homography matches: ' + str(len(good_model_pts)) + ')', model_img, input_img, kp_pairs, mask, H, show_win)

    # matchesMask, input_image_homo, good, model_pts, input_pts, M, M2
    return(mask, good_model_pts, good_input_pts, H, H2)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None, show_win= True):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    if show_win:
        filename = win.split('|')[0] + ".jpg"
        cv2.imshow(win, vis)
        #cv2.imwrite(filename, vis)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]

            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i][0]]
                cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis

def validate_homography(perspective_trans_matrix):
    ## 1. Determinant of 2x2 matrix  (source: https://dsp.stackexchange.com/questions/17846/template-matching-or-object-recognition)
    # Property of an affine (and projective?) transformation:  (source: http://answers.opencv.org/question/2588/check-if-homography-is-good/)
    #   -If the determinant of the top-left 2x2 matrix is > 0 the transformation is orientation-preserving.
    #   -Else if the determinant is < 0, it is orientation-reversing.
    det = perspective_trans_matrix[0, 0] * perspective_trans_matrix[1, 1] - perspective_trans_matrix[0, 1] * \
                                                                            perspective_trans_matrix[1, 0]
    #print("----- determinant of topleft 2x2 matrix: ", det)
    if det < 0:
        #print("determinant<0, homography unvalid")
        # exit()
        return False

    ## 2. copied from https://dsp.stackexchange.com/questions/17846/template-matching-or-object-recognition
    # other explanation: https://dsp.stackexchange.com/questions/1990/filtering-ransac-estimated-homographies
    N1 = (perspective_trans_matrix[0, 0] * perspective_trans_matrix[0, 0] + perspective_trans_matrix[0, 1] *
          perspective_trans_matrix[0, 1]) ** 0.5
    if N1 > 4 or N1 < 0.1:
        #print("not ok 1")
        return False
        # exit()
    N2 = (perspective_trans_matrix[1, 0] * perspective_trans_matrix[1, 0] + perspective_trans_matrix[1, 1] *
          perspective_trans_matrix[1, 1]) ** 0.5
    if N2 > 4 or N2 < 0.1:
        #print("not ok 2")
        return False
        # exit()
    N3 = (perspective_trans_matrix[2, 0] * perspective_trans_matrix[2, 0] + perspective_trans_matrix[2, 1] *
          perspective_trans_matrix[2, 1]) ** 0.5
    if N3 > 0.002:
        #print("not ok 3")
        return False
        # exit()

    return True

def max_euclidean_distance(model, transformed_input):

    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return max(euclidean_distance)

def euclidean_distance(model, transformed_input):
    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return euclidean_distance



# ---- vanaf hier voooooral oude rommel

def sift_detect_and_compute(image):
    # --------- SIFT FEATURE DETETCION & DESCRIPTION ------------------------
    # Initiate SIFT detector  # TODO: what is best? ORB SIFT  &&& FLANN?
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_model, des_model = sift.detectAndCompute(image,None)  # returns keypoints and descriptors
    return (kp_model, des_model)

def orb_detect_and_compute(image):

    '''detector = cv2.ORB_create()
    kp_scene = detector.detect(image)
    k_scene, d_scene = detector.compute(image, kp_scene)'''
    orb = cv2.ORB_create()

    kp, des = orb.detectAndCompute(image, None)
    des = np.float32(des)

    return (kp, des)

def flann_matching(des_model, des_input, kp_model, kp_input, model_image, input_image):
    # --------- FEATURE MATCHING : FLANN MATCHER -------------------
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)   # voor met SIFT

    FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,    # voor met ORB
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2

    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_model,des_input,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.80*n.distance:
            good.append(m)

    print("aantal matches= ", len(good))

    if len(good)>MIN_MATCH_COUNT:
        model_pts = np.float32([ kp_model[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        input_pts = np.float32([ kp_input[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Checken in andere kant

        # Find only the good, corresponding points (lines of matching points may not cross)
        # Returns the perspective transformation matrix M
        M, mask = cv2.findHomography(model_pts, input_pts, cv2.RANSAC,5.0)  #tresh : 5
        #M, mask = cv2.findHomography(input_pts, model_pts, cv2.RANSAC, 5.0)  # tresh : 5

        matchesMask = mask.ravel().tolist()
        # TODO wat als model_image en input_image niet zelfde resolutie hebben?
        h,w = model_image.shape

        print("aantal good matches: " , matchesMask.count(1))

        # the square that's drawn on the model. Just the prerspective transformation of the model image contours
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #print("###points: ", pts)
        dst = cv2.perspectiveTransform(pts,M)

        #Homography RANSAC is used to reject outliers
        M2, mask2 = cv2.findHomography(input_pts, model_pts, cv2.RANSAC, 5.0)  # we want to transform the input-plane to the model-plane
        h2,w2 = input_image.shape
        '''
        perspective_transform_input = cv2.warpPerspective(input_image, M2, (w2, h2 ))
        plt.figure()
        plt.subplot(131), plt.imshow(model_image), plt.title('Model')
        plt.subplot(132), plt.imshow(perspective_transform_input), plt.title('Perspective transformed Input')
        plt.subplot(133), plt.imshow(input_image), plt.title('Input')
        plt.show(block=False)
        '''
        input_image_homo = cv2.polylines(input_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)  # draw homography square
        #model_image_homo = cv2.polylines(model_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  # draw homography square


        return (matchesMask, input_image_homo, good, model_pts, input_pts, M, M2)
        #return (matchesMask, model_image_homo, good, model_pts, input_pts)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None

def plot_features(clustered_model_features, clustered_input_features, one_building, model_image, input_image):
    if one_building:  # Take the first found homography, no more computations needed
        # Reduce dimensions
        print("one building only")
        plt.figure()
        plt.scatter(clustered_model_features[:, 0], clustered_model_features[:, 1])
        # plt.scatter(model_center[:,0],model_center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Width'), plt.ylabel('Height')
        plt.imshow(model_image)
        plt.show(block=False)

        plt.figure()
        plt.scatter(clustered_input_features[:, 0], clustered_input_features[:, 1])
        # plt.scatter(input_center[:,0],input_center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Width'), plt.ylabel('Height')
        #plt.imshow(cv2.imread('img/' + input_name + '.' + img_tag))
        plt.imshow(input_image)
        plt.show(block=False)


    else: # More than one building
        plt.figure()
        for feat in clustered_model_features:
            plt.scatter(feat[:, 0], feat[:, 1], c=np.random.rand(3,), s=5)
        plt.xlabel('Width'), plt.ylabel('Height')
        plt.imshow(model_image)
        plt.show(block=False)

        plt.figure()
        for feat in clustered_input_features:
            plt.scatter(feat[:, 0], feat[:, 1], c=np.random.rand(3,), s=5)
        plt.xlabel('Width'), plt.ylabel('Height')
        plt.imshow(input_image)
        plt.show(block=False)


    return None

