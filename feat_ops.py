import cv2
import numpy as np
import affine_transformation as at
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from common import anorm, getsize
import normalising
import prepocessing

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
    print('matching...')
    raw_matches = matcher.knnMatch(desc_model, trainDescriptors = desc_input, k = 2) #2

    # p_model and p_input are the 'first-good' matches
    p_model, p_input, kp_pairs = filter_matches(kp_model, kp_input, raw_matches)

    if len(p_model) >= MIN_MATCH_COUNT:
        # Returns the perspective transformation matrix M: transformaion from model to inputplane
        H, mask = cv2.findHomography(p_model, p_input, cv2.RANSAC, 5.0)

        # NOTE!! THIS is persp trans from inputplane to modelplane!! used for perspective elimination
        H2, mask2 = cv2.findHomography(p_input, p_model, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(mask), len(mask)))

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

        print("Raw matches: ", len(p_model))
        print("Perspective matches: ", len(good_model_pts))

    else:
        H, mask = None, None
        print('%d matches found, not enough for homography estimation' % len(p_model))

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
        print("klalalal    ", filename)
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
        print("determinant<0, homography unvalid")
        # exit()
        return False

    ## 2. copied from https://dsp.stackexchange.com/questions/17846/template-matching-or-object-recognition
    # other explanation: https://dsp.stackexchange.com/questions/1990/filtering-ransac-estimated-homographies
    N1 = (perspective_trans_matrix[0, 0] * perspective_trans_matrix[0, 0] + perspective_trans_matrix[0, 1] *
          perspective_trans_matrix[0, 1]) ** 0.5
    if N1 > 4 or N1 < 0.1:
        print("not ok 1")
        return False
        # exit()
    N2 = (perspective_trans_matrix[1, 0] * perspective_trans_matrix[1, 0] + perspective_trans_matrix[1, 1] *
          perspective_trans_matrix[1, 1]) ** 0.5
    if N2 > 4 or N2 < 0.1:
        print("not ok 2")
        return False
        # exit()
    N3 = (perspective_trans_matrix[2, 0] * perspective_trans_matrix[2, 0] + perspective_trans_matrix[2, 1] *
          perspective_trans_matrix[2, 1]) ** 0.5
    if N3 > 0.002:
        print("not ok 3")
        return False
        # exit()

    return True

def perspective_correction(H2, p_model, p_input, model_pose_features, input_pose_features, model_img, input_img, plot=False):
    # we assume input_pose and model_pose contain same amount of features, as we would also expected in this stage of pipeline

    h, w = input_img.shape
    h = round(h * 6 / 5)
    w = round(w * 6 / 5)

    perspective_transform_input = cv2.warpPerspective(input_img, H2, (w, h)) # persp_matrix2 transforms input onto model_plane
    if plot:
        plt.figure()
        plt.subplot(221), plt.imshow(model_img), plt.title('Model')
        plt.subplot(222), plt.imshow(perspective_transform_input), plt.title('Perspective transformed Input')
        plt.subplot(223), plt.imshow(input_img), plt.title('Input')
        plt.show(block=False)

    my_input_pts2 = np.float32(p_input).reshape(-1, 1, 2)  # bit reshapeing so the cv2.perspectiveTransform() works
    p_input_persp_trans = cv2.perspectiveTransform(my_input_pts2, H2)  # transform(input_pts_2D)
    p_input_persp_trans = np.squeeze(p_input_persp_trans[:])  # strip 1 dimension

    max_euclidean_error = max_euclidean_distance(p_model, p_input_persp_trans)
    print('@@@@ PERSSPECTIVE 1: max error: ', max_euclidean_error)

    #TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = normalising.feature_scaling(p_model)
    input_features_trans_norm = normalising.feature_scaling(p_input_persp_trans)

    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print('@@@@ PERSSPECTIVE NORM 1: max error: ', max_euclidean_error)

    # -- 2E MANIERRR: normaliseren enkel de pose
    input_pose_trans = p_input_persp_trans[len(p_input_persp_trans) - len(input_pose_features): len(
        p_input_persp_trans)]  # niet perse perspective corrected, hangt af van input
    model_pose_norm = normalising.feature_scaling(model_pose_features)
    input_pose_trans_norm = normalising.feature_scaling(input_pose_trans)

    max_euclidean_error = max_euclidean_distance(model_pose_norm, input_pose_trans_norm)

    print('@@@@ PERSSPECTIVE NORM 2: max error: ', max_euclidean_error)

    markersize = 3
    # model_img_arr = np.asarray(model_img)
    # input_img_arr = np.asarray(input_img)
    # input_persp_img_arr = np.asarray()

    if plot:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
        #ax1.imshow(model_img)
        ax1.imshow(np.asarray(model_img), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("model")
        ax1.plot(*zip(*p_model), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose_features), marker='o', color='red', ls='', label='pose', ms=markersize )  # ms = markersize
        red_patch = mpatches.Patch(color='magenta', label='model')
        ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("input")
        ax2.imshow(np.asarray(input_img), cmap='gray')
        ax2.plot(*zip(*p_input), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose_features), marker='*', color='r', ls='', ms=markersize)
        ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("persp corr input (features+pose)")
        ax3.imshow(np.asarray(perspective_transform_input), cmap='gray')
        ax3.plot(*zip(*input_pose_trans), marker='o', color='b', ls='', ms=markersize)
        ax3.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])

        ax4.set_title("trans-input onto model")
        ax4.imshow(np.asarray(model_img), cmap='gray')
        ax4.plot(*zip(*p_input_persp_trans), marker='o', color='b', ls='', ms=markersize)
        ax4.plot(*zip(*p_model), marker='o', color='magenta', ls='', ms=markersize)
        ax4.plot(*zip(*model_pose_features), marker='o', color='green', ls='', ms=markersize)
        ax4.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])
        # plt.tight_layout()
        plt.show(block=False)

    return (p_input_persp_trans, input_pose_trans,  perspective_transform_input)

def affine_trans_interaction_both(p_model_good, p_input_good, model_pose, input_pose,  model_img, input_img, label):
    #input_pose = p_input_good[len(p_input_good) - size_pose: len(p_input_good)]  # niet perse perspective corrected, hangt af van input
    #model_pose = p_model_good[len(p_model_good) - size_pose: len(p_input_good)]

    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_pose)

    (input_transformed_torso, M_tor) = at.find_transformation(np.vstack((p_model_good, model_torso)),np.vstack((p_input_good, input_torso)))
    (input_transformed_legs, M_legs) = at.find_transformation(np.vstack((p_model_good, model_legs)),np.vstack((p_input_good, input_legs)))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_torso)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error)
    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_legs)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM" + label + "  error_legs: ", max_euclidean_error)


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_img), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_img), cmap='gray')
    ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split" + label)
    ax3.imshow(np.asarray(model_img), cmap='gray')
    ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*np.vstack((input_transformed_torso, input_transformed_legs))), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input torso'),
                        mpatches.Patch(color='magenta', label='model')])

    # plt.tight_layout()
    plt.show(block=False)
    return None

# enkel A berekenen uit pose features lijkt mij het logischte want enkel de pose kan varieeren in ratio
# de scene niet aangezien die ratio's normaal vast zijn!!
def affine_trans_interaction_only_pose(p_model_good, p_input_good, model_pose, input_pose, model_img, input_img, label):
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    #model_torso = np.vstack((model_torso, p_model_good[0], p_model_good[1], p_model_good[10] ))
    #input_torso = np.vstack((input_torso, p_input_good[0], p_input_good[1], p_input_good[10]))

    #model_legs = np.vstack((model_legs, p_model_good[0], p_model_good[1], p_model_good[10] ))
    #input_legs = np.vstack((input_legs, p_input_good[0], p_input_good[1], p_input_good[10]))


    (input_transformed_torso, M_tor) = at.find_transformation(model_torso,input_torso)
    (input_transformed_legs, M_legs) = at.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((p_input_good, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((p_input_good, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_torso)

    max_euclidean_error_torso = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error_torso)


    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_legs)

    max_euclidean_error_legs = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_legs: ", max_euclidean_error_legs)

    if max_euclidean_error_torso < 0.15 and max_euclidean_error_legs < 0.15:
        print("#### MATCH!!!  ###")
    else:
        print("#### NO MATCH!! ###")


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 2
    ms_pose = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_img), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=ms_pose)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_img), cmap='gray')
    ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=ms_pose)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split " + label)
    ax3.imshow(np.asarray(model_img), cmap='gray')
    ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed_legs), marker='o', color='green', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed_torso), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input torso'),
                        mpatches.Patch(color='magenta', label='model')])

    # plt.tight_layout()
    plt.show(block=False)
    return None

def affine_trans_interaction_pose_rand_scene(p_model_good, p_input_good, model_pose, input_pose,  model_img, input_img, label, plot=False):
    # TODO: Deze geeft (momenteel betere resultaten dan den _normalise => verschillen tussen matches en niet-matches is pak groter
    #TODO: normalising van whole !! en niet normaliseren van legs en torso appart
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    model_torso = np.vstack((model_torso, p_model_good[0], p_model_good[1], p_model_good[10] ))
    input_torso = np.vstack((input_torso, p_input_good[0], p_input_good[1], p_input_good[10]))

    model_legs = np.vstack((model_legs, p_model_good[0], p_model_good[1], p_model_good[10] ))
    input_legs = np.vstack((input_legs, p_input_good[0], p_input_good[1], p_input_good[10]))


    (input_transformed_torso, M_tor) = at.find_transformation(model_torso, input_torso)
    (input_transformed_legs, M_legs) = at.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((p_input_good, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((p_input_good, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_torso)

    max_euclidean_error_torso_norm = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE RAND NORM " + label + "  error_torso: ", max_euclidean_error_torso_norm)

    model_features_norm = normalising.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = normalising.feature_scaling(input_transformed_legs)

    max_euclidean_error_legs_norm = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE RAND NORM " + label + "  error_legs: ", max_euclidean_error_legs_norm)

    if max_euclidean_error_torso_norm < 0.15 and max_euclidean_error_legs_norm < 0.15:
        print("#### MATCH!!!  ###")
        match = True
    else:
        print("#### NO MATCH!! ###")
        match = False

    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    print("#### AFFINE RAND" + label + "  error_torso: ", max_euclidean_error_torso)
    print("#### AFFINE RAND" + label + "  error_legs: ", max_euclidean_error_legs)


    markersize = 3
    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        implot = ax1.imshow(np.asarray(model_img), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("model")
        ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
                 ms=markersize)  # ms = markersize
        red_patch = mpatches.Patch(color='magenta', label='model')
        ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("input")
        ax2.imshow(np.asarray(input_img), cmap='gray')
        ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("aff split() " + label)
        ax3.imshow(np.asarray(model_img), cmap='gray')
        ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='',
                 label='model',
                 ms=markersize)  # ms = markersize
        ax3.plot(*zip(*input_transformed_legs), marker='o', color='green', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax3.plot(*zip(*input_transformed_torso), marker='o', color='blue', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax3.legend(handles=[mpatches.Patch(color='blue', label='trans-input torso'),
                            mpatches.Patch(color='green', label='trans-input legs'),
                            mpatches.Patch(color='magenta', label='model')])

        # plt.tight_layout()
        plt.show(block=False)
    return (match, max_euclidean_error_torso_norm, max_euclidean_error_legs_norm)



def max_euclidean_distance(model, transformed_input):

    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return max(euclidean_distance)


# ---- vanaf hier voooooral oude rommel

# mag weg
def affine_trans_interaction_pose_rand_scene_normalise(p_model_good, p_input_good, model_pose, input_pose,  model_img, input_img, label):

    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    amount_of_random_background_features = 3
    model_both = np.vstack((model_pose, p_model_good[0], p_model_good[1], p_model_good[10]))
    input_both = np.vstack((input_pose, p_input_good[0], p_input_good[1], p_input_good[10]))

    # Normalise the whole (all features together = pose- and background features)
    model_both_norm = normalising.feature_scaling(model_both)
    input_both_norm = normalising.feature_scaling(input_both)

    # Extract the normalised poses and normalised background features
    model_pose_norm = model_both_norm[0: len(model_both_norm) - amount_of_random_background_features]
    input_pose_norm = input_both_norm[0: len(input_both_norm) - amount_of_random_background_features]

    model_bf_norm = model_both_norm[len(model_both_norm) - amount_of_random_background_features: len(model_both_norm)]
    input_bf_norm = input_both_norm[len(input_both_norm) - amount_of_random_background_features: len(input_both_norm)]

    # Split the normalised pose
    (model_face_norm, model_torso_norm, model_legs_norm) = prepocessing.split_in_face_legs_torso(model_pose_norm)
    (input_face_norm, input_torso_norm, input_legs_norm) = prepocessing.split_in_face_legs_torso(input_pose_norm)


    # Calc affine transformation from combination of (background features + torso) & (background features + legs)
    (input_transformed_torso_norm, M_tor) = at.find_transformation(np.vstack((model_torso_norm, model_bf_norm)), np.vstack((input_torso_norm, input_bf_norm)))
    (input_transformed_legs_norm, M_legs) = at.find_transformation(np.vstack((model_legs_norm, model_bf_norm)), np.vstack((input_legs_norm, input_bf_norm)))

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]

    # TODO: Calc eucl distance only of 3 random background features? Or from all detected features ???
    #input_transformed_torso = unpad(np.dot(pad(np.vstack((p_input_good, input_torso))), M_tor))
    #input_transformed_legs = unpad(np.dot(pad(np.vstack((p_input_good, input_legs))), M_legs))

    max_euclidean_error_torso = max_euclidean_distance(np.vstack((model_torso_norm, model_bf_norm)), input_transformed_torso_norm)
    print("#### AFFINE RAND NORM " + label + "  error_torso: ", max_euclidean_error_torso)

    max_euclidean_error_legs = max_euclidean_distance(np.vstack((model_legs_norm, model_bf_norm)), input_transformed_legs_norm)
    print("#### AFFINE RAND NORM " + label + "  error_legs: ", max_euclidean_error_legs)

    if max_euclidean_error_torso < 0.15 and max_euclidean_error_legs < 0.15:
        print("#### MATCH!!!  ###")
    else:
        print("#### NO MATCH!! ###")

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


def affine_transform_urban_scene_and_pose_OLD(one_building, model_pose_features, input_pose_features, clustered_input_features, clustered_model_features,
                                          model_image, input_image, perspective_trans_matrix):
    # -------------  CALC AFFINE TRANSFORMATION  ------------------##
    # Calc affine trans between the wrest points and some random feature points of the building
    # The question is: WHICH feature points should we take??
    # An option is to go for the "best matches" (found during featuring-matching)
    # An other option is just to take an certain number of random matches

    # Third option would be to take all the building feature points,
    # but that would probably limit transformation in aspect of the mutual spatial
    # relation between the person and the building
    # TODO: other options??

    object_index = 0  # TODO make algo that decides which object to take (the one without the person)
    # TODO: moet niet perse door een algoritme worden bepaalt, kan ook eigenschap zijn van urban scene en dus gemarkt door mens

    if not one_building:
        clustered_input_features = clustered_input_features[object_index]
        clustered_model_features = clustered_model_features[object_index]

    #### CALC AFFINE TRANSFORMATION OF WHOLE  (building feature points + person keypoints) #############"
    '''
    input_pose_features = np.append(input_pose_features, [clustered_input_features[0]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[2]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[3]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[4]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[8]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[9]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[11]], 0)


    model_pose_features = np.append(model_pose_features, [clustered_model_features[0]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[2]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[3]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[4]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[8]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[9]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[11]], 0)
    '''
    #clustered_model_features = np.append(clustered_model_features, [model_pose_features[0]], 0)
    #clustered_model_features = np.append(clustered_model_features, [model_pose_features[1]], 0)

    #clustered_input_features = np.append(clustered_input_features, [input_pose_features[0]], 0)
    #clustered_input_features = np.append(clustered_input_features, [input_pose_features[1]], 0)

    # Calc transformation of whole (building feature points + person keypoints)
    #(input_transformed, transformation_matrix) = affine_transformation.find_transformation(model_pose_features,input_pose_features)
    (input_transformed, transformation_matrix) = at.find_transformation(clustered_model_features,
                                                                                           clustered_input_features)
    print("affine matrix: ", transformation_matrix)

    #####################################################################################




    ### CALC FIRST AFFINE TrANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    # Some random selected features (of the buidling)
    #input_building_features = np.array([clustered_input_features[0], clustered_input_features[2], clustered_input_features[6], clustered_input_features[14]])
    #model_building_features = np.array([clustered_model_features[0], clustered_model_features[2], clustered_model_features[6], clustered_model_features[14]])

    # Calc the transformation matrix
    #(input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(model_building_features,input_building_features)

    # Calc the transformed features again with same transformation matrix
    # But now with the people keypoints appended

    #pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #unpad = lambda x: x[:, :-1]
    #transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #input_transformed = transform(input_pose_features)



    # ------------------ 2 BUILDINGS --------------------------

    #     ### EERSTE MANIER: calc affine trans of whole (building features + person key-points)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(whole_output_features,
    #                                                                                            whole_input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### TWEEDE MANIER: CALC FIRST AFFINE TANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     # input_building_features = np.array(
    #     #     [clustered_input_features[object_index][0], clustered_input_features[object_index][2], clustered_input_features[object_index][6],
    #     #      clustered_input_features[object_index][14]])
    #     # model_building_features = np.array(
    #     #     [clustered_model_features[object_index][0], clustered_model_features[object_index][2], clustered_model_features[object_index][6],
    #     #      clustered_model_features[object_index][14]])
    #     #
    #     # # Calc the transformation matrix of building features
    #     # (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #     #     model_building_features,
    #     #     input_building_features)
    #
    #
    #
    #     #3e MANIER:  Calc transformation of person-features
    #     input_features = np.append(input_features, clustered_input_features[object_index], 0)
    #     output_features = np.append(output_features, clustered_model_features[object_index], 0)
    #
    #     #print("ejejeje: " , input_features)
    #
    #
    #
    #     (input_person_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #         output_features,
    #         input_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #     img = cv2.imread('img/' + model_name + '.' + img_tag)
    #     rows, cols, ch = img.shape
    #
    #     input_features = input_features.astype(np.float32)
    #     output_features = output_features.astype(np.float32)
    #     pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    #     pts2 = output_features #np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    #
    #     print("pts: ", pts1)
    #
    #
    #     print("whoooole" , input_features)
    #     '''
    #     print("whoooole", output_features)
    #     print("shape: " , input_features.shape)
    #     print("shape2; " , output_features.shape)
    #     '''

    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*clustered_model_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(input_image)
    ax2.plot(*zip(*clustered_input_features), marker='o', color='r', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("affine transformation (features+pose)")
    ax3.imshow(model_image)
    ax3.plot(*zip(*clustered_model_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
                        mpatches.Patch(color='magenta', label='model')])
    #plt.tight_layout()
    plt.show(block=False)
    return None

    # cv2.waitKey(0)

    # if one_building: # Take the first found homography, no more computations needed
    #     #Reduce dimensions
    #     clustered_model_features = np.squeeze(clustered_features[0])
    #     clustered_input_features = np.squeeze(clustered_features[1])
    #     print("one building only")
    #
    #     plt.scatter(clustered_model_features[:,0],clustered_model_features[:,1])
    #     #plt.scatter(model_center[:,0],model_center[:,1],s = 80,c = 'y', marker = 's')
    #     plt.xlabel('Width'),plt.ylabel('Height')
    #     plt.imshow(model_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     plt.scatter(clustered_input_features[:,0],clustered_input_features[:,1])
    #     #plt.scatter(input_center[:,0],input_center[:,1],s = 80,c = 'y', marker = 's')
    #     plt.xlabel('Width'),plt.ylabel('Height')
    #     plt.imshow(cv2.imread('img/' + input_name + '.' + img_tag))
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     #-------------  CALC AFFINE TRANSFORMATION  ------------------##
    #     # Calc affine trans between the wrest points and some random feature points of the building
    #     # The question is: WHICH feature points should we take??
    #     # An option is to go for the "best matches" (found during featuring-matching)
    #     # An other option is just to take an certain number of random matches
    #
    #     # Third option would be to take all the building feature points,
    #     # but that would probably limit transformation in aspect of the mutual spatial
    #     # relation between the person and the building
    #     # TODO: other options??
    #
    #     # Create feature array for first person
    #     # feature points of pisa tower are in A
    #     # feautes van pols =
    #
    #     # p9_r_pols = np.array([[152, 334]])  #pisa9
    #     # p9_l_pols = np.array([[153,425]])
    #     # p10_r_pols = np.array([[256, 362]])   #pisa10
    #     # p10_l_pols = np.array([[247, 400]])
    #
    #     input_features = np.array([[152, 334], [153, 425]])  #pisa9
    #     output_features = np.array([[256, 362], [247, 400]]) #pisa10
    #
    #
    #
    #     input_features =  np.array([[391,92]])  #taj3  enkel recher pols
    #     input_features = np.array([[463, 89]]) # foute locatie
    #     input_features = np.array([[391, 92], [517, 148]])  # taj3  enkel recher pols + nek
    #     input_features = np.array([[391, 92], [429, 126]])  # taj3  enkel recher pols + r elbow
    #
    #     output_features = np.array([[303,37]]) #taj4 enkel rechter pols
    #     output_features = np.array([[303, 37],[412, 90]])  # taj4 enkel rechter pols + nek
    #     output_features = np.array([[303, 37], [347, 70]])  # taj4 enkel rechter pols + r elbow
    #
    #     #### CALC AFFINE TRANSFORMATION OF WHOLE  (building feature points + person keypoints) #############"
    #
    #     input_features = np.append(input_features, [clustered_input_features[0]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[2]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[6]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[14]], 0)
    #
    #     output_features = np.append(output_features, [clustered_model_features[0]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[2]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[6]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[14]], 0)
    #
    #
    #     # Calc transformation of whole (building feature points + person keypoints)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(output_features,
    #                                                                                            input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### CALC FIRST AFFINE TrANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     input_building_features = np.array([clustered_input_features[0], clustered_input_features[2], clustered_input_features[6], clustered_input_features[14]])
    #     model_building_features = np.array([clustered_model_features[0], clustered_model_features[2], clustered_model_features[6], clustered_model_features[14]])
    #
    #
    #
    #     # Calc the transformation matrix
    #     (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(model_building_features,
    #                                                                                            input_building_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #
    #
    #
    #     markersize = 3
    #
    #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    #     implot = ax1.imshow(model_image)
    #     # ax1.set_title(model_image_name + ' (model)')
    #     ax1.set_title("model")
    #     ax1.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     red_patch = mpatches.Patch(color='magenta', label='model')
    #     ax1.legend(handles=[red_patch])
    #
    #     # ax2.set_title(input_image_name + ' (input)')
    #     ax2.set_title("input")
    #     ax2.imshow(input_image)
    #     ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
    #     ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
    #
    #     ax3.set_title("transformation")
    #     ax3.imshow(model_image)
    #     ax3.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    #     ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
    #                         mpatches.Patch(color='magenta', label='model')])
    #     #plt.tight_layout()
    #     plt.show(block=False)
    #
    #     plt.figure()
    #
    #
    #
    #
    #
    # else: # More than one building
    #     for feat in clustered_model_features:
    #         plt.scatter(feat[:, 0], feat[:, 1])
    #     plt.xlabel('Height'),plt.ylabel('Weight')
    #     plt.imshow(model_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #
    #     for feat in clustered_input_features:
    #         plt.scatter(feat[:, 0], feat[:, 1])
    #     plt.xlabel('Height'),plt.ylabel('Weight')
    #     plt.imshow(input_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     # -------------  CALC AFFINE TRANSFORMATION  ------------------##
    #
    #     # p9_r_pols = np.array([[152, 334]])  #pisa9
    #     # p9_l_pols = np.array([[153,425]])
    #     # p10_r_pols = np.array([[256, 362]])   #pisa10
    #     # p10_l_pols = np.array([[247, 400]])
    #
    #     input_features = np.array([[256, 362], [247, 400]], np.float32)   # pisa9
    #     output_features = np.array([[152, 334], [153, 425]], np.float32) # pisa10
    #
    #
    #     input_features = np.array([[127, 237], [206, 234], [317, 205] ], np.float32)  # trap1
    #     #input_features = np.array([[218, 299], [280, 300]])  # trap3
    #     #input_features = np.array([[136, 230], [297, 536], [343, 542]])  #trap9  rpols, renkel, lenkel
    #     input_features = np.array([[113, 290], [179, 290]], np.float32)  # trap1
    #
    #     output_features = np.array([[116, 289], [188, 284], [307, 257]], np.float32)  # trap2
    #     #output_features = np.array([[150, 230],[319, 570], [376, 587]], np.float32) #trap8   rpols, renkel, lenkel
    #     #output_features = np.array([[254, 248], [293, 253]], np.float32)  # trap4
    #     output_features = np.array([[127, 237], [206, 234]], np.float32)  # trap1
    #
    #     object_index = 1
    #
    #     whole_input_features = np.append(input_features, [clustered_input_features[object_index][0]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][2]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][6]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][8]], 0)
    #
    #     whole_output_features = np.append(output_features, [clustered_model_features[object_index][0]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][2]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][6]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][8]], 0)
    #
    #     ### EERSTE MANIER: calc affine trans of whole (building features + person key-points)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(whole_output_features,
    #                                                                                            whole_input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### TWEEDE MANIER: CALC FIRST AFFINE TANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     # input_building_features = np.array(
    #     #     [clustered_input_features[object_index][0], clustered_input_features[object_index][2], clustered_input_features[object_index][6],
    #     #      clustered_input_features[object_index][14]])
    #     # model_building_features = np.array(
    #     #     [clustered_model_features[object_index][0], clustered_model_features[object_index][2], clustered_model_features[object_index][6],
    #     #      clustered_model_features[object_index][14]])
    #     #
    #     # # Calc the transformation matrix of building features
    #     # (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #     #     model_building_features,
    #     #     input_building_features)
    #
    #
    #
    #     #3e MANIER:  Calc transformation of person-features
    #     input_features = np.append(input_features, clustered_input_features[object_index], 0)
    #     output_features = np.append(output_features, clustered_model_features[object_index], 0)
    #
    #     #print("ejejeje: " , input_features)
    #
    #
    #
    #     (input_person_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #         output_features,
    #         input_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #     img = cv2.imread('img/' + model_name + '.' + img_tag)
    #     rows, cols, ch = img.shape
    #
    #     input_features = input_features.astype(np.float32)
    #     output_features = output_features.astype(np.float32)
    #     pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    #     pts2 = output_features #np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    #
    #     print("pts: ", pts1)
    #
    #
    #     print("whoooole" , input_features)
    #     '''
    #     print("whoooole", output_features)
    #     print("shape: " , input_features.shape)
    #     print("shape2; " , output_features.shape)
    #     '''
    #
    #
    #
    #
    #     markersize = 3
    #
    #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    #     implot = ax1.imshow(model_image)
    #     # ax1.set_title(model_image_name + ' (model)')
    #     ax1.set_title("model")
    #     ax1.plot(*zip(*whole_output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     red_patch = mpatches.Patch(color='magenta', label='model')
    #     ax1.legend(handles=[red_patch])
    #
    #     # ax2.set_title(input_image_name + ' (input)')
    #     ax2.set_title("input")
    #     ax2.imshow(input_image)
    #     ax2.plot(*zip(*whole_input_features), marker='o', color='r', ls='', ms=markersize)
    #     ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
    #
    #     ax3.set_title("transformation")
    #     ax3.imshow(model_image)
    #     ax3.plot(*zip(*whole_output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    #     ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
    #                         mpatches.Patch(color='magenta', label='model')])
    #     # plt.tight_layout()
    #     plt.show(block=False)
    #
    #     plt.figure()
    #
    #
    #
    # plt.show()
    # cv2.waitKey(0)


