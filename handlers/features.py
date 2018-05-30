import logging
import cv2
import numpy as np
from matplotlib import pyplot as plt

from handlers import function
import thresholds



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
            flann_params = dict(algorithm=thresholds.FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=thresholds.FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv2.FlannBasedMatcher(flann_params, search_params)  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = thresholds.FILTER_RATIO):
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

def find_homography(win, matcher, desc_model, desc_input, kp_model, kp_input, model_img, input_img, show_win):
    #print('matching...')
    raw_matches = matcher.knnMatch(desc_model, trainDescriptors = desc_input, k = 2) #2

    # p_model and p_input are the 'first-good' matches
    p_model, p_input, kp_pairs = filter_matches(kp_model, kp_input, raw_matches)

    if len(p_model) < thresholds.MIN_MATCH_COUNT:
        logging.debug('%d matches found, not enough for homography estimation' % len(p_model))
        return (None, None, None, None, None, None)

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



    # Render nice window with nice view of matches
    #_vis = explore_match(win + '| (raw matches: ' + str(len(p_model)) + '  homography matches: ' + str(len(good_model_pts)) + ')', model_img, input_img, kp_pairs, mask, H, show_win)

    # matchesMask, input_image_homo, good, model_pts, input_pts, M, M2
    return(mask, good_model_pts, good_input_pts, H, H2, len(good_model_pts))

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
            m = (function.anorm(np.array(p1) - (x, y)) < r) | (function.anorm(np.array(p2) - (x, y)) < r)
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
