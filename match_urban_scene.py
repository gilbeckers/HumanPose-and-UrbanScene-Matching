'''
Feature-based urban scene matching.
uses find_obj.py

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import feat_ops
import numpy as np
from matplotlib import pyplot as plt
import parse_openpose_json
import affine_transformation
from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
MIN_MATCH_COUNT = 4

list_poses = {
    # 1: linkerpols   2:rechterpols
    "blad1": np.array([[113, 290], [179, 290]]),
    "notredam1": np.array([[113, 290], [179, 290]]),
    "notredam3": np.array([[113, 290], [179, 290]]),
    "pisa_chapel": np.array([[113, 290], [179, 290]]),
    "blad2": np.array([[113, 290], [179, 290]]),
    "trap1": np.array([[173,444],[278, 440]]),  #np.array([[113, 290], [179, 290]], np.float32)  # trap1
    "trap2": np.array([[197,361],[302, 355]]),  # np.array([[127, 237], [206, 234]], np.float32)  # trap1
    "trap3": np.array([[333,448], [423,452]]),
    "trap4": np.array([[254, 248], [293, 253]]),
    "trap7": np.array([[254, 248], [293, 253]]),
    "trap8": np.array([[150, 230],[225, 225]]),  #trap8   rpols, renkel, lenkel
    "trap9": np.array([[136, 230], [217, 221]]),
    "taj1": np.array([[391, 92], [429, 126]]),
    "taj2": np.array([[391, 92], [429, 126]]),#trap9  rpols, renkel, lenkel  , [343, 542]
    "taj3": np.array([[302,36], [351, 72]]),  # taj3  enkel recher pols + r elbow   [391, 92], [429, 126]
    "taj4": np.array([[389,89], [425, 123]]),   # taj4 enkel rechter pols + r elbow     #np.array([[303, 37], [412, 90]])  # taj4 enkel rechter pols + nek
    "pisa2": np.array([[152, 334], [146, 419]]),
    "pisa3": np.array([[152, 334], [146, 419]]),
    "pisa4": np.array([[152, 334], [146, 419]]),
    "pisa5": np.array([[152, 334], [146, 419]]),
    "pisa6": np.array([[152, 334], [146, 419]]),
    "pisa7": np.array([[152, 334], [146, 419]]),
    "pisa8": np.array([[152, 334], [146, 419]]),
    "pisa9": np.array([[152, 334], [146, 419]]),  #  np.array([[256, 362], [247, 400]], np.float32)
    "pisa10" : np.array([[256, 362], [247, 400]]),
    "pisa12": np.array([[152, 334], [146, 419]]),
    "pisa101" : np.array([[217, 353], [209, 404]]),
    "foto3" : np.array([[217, 353], [209, 404]]),
    "foto4" : np.array([[217, 353], [209, 404]])
    }

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    '''opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk')
    try:
        fn1, fn2 = args
    except:
        fn1 = '../data/box.png'
        fn2 = '../data/box_in_scene.png'''

    # combination of detector(orfb, surf, sift, brief) and matcher (flann, bruteforce)
    feature_name = 'sift-flann'

    model_name = 'trap1.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1
    input_name = 'trap5.jpg'  # goeie : "pisa10"  taj4  # trap2     trap3
    model_image = cv2.imread('img/' + model_name, 0)
    input_image = cv2.imread('img/' + input_name, 0)

    # Read poses
    #model_pose_features = list_poses[model_name.split('.')[0]]
    #input_pose_features = list_poses[input_name.split('.')[0]]

    model_pose_features = parse_openpose_json.parse_JSON_single_person('json_data/' + model_name.split('.')[0] + '.json')
    input_pose_features = parse_openpose_json.parse_JSON_single_person('json_data/' + input_name.split('.')[0] + '.json')
    assert model_pose_features.shape == input_pose_features.shape

    detector, matcher = feat_ops.init_feature(feature_name)

    if model_image is None:
        print('Failed to load fn1:', model_name)
        sys.exit(1)

    if input_image is None:
        print('Failed to load fn2:', input_image)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    # ---------- STEP 1: FEATURE DETECTION AND DESCRIPTION (ORB, SIFT, SURF, BRIEF, ASIFT --------------------
    kp_model, desc_model = detector.detectAndCompute(model_image, None)
    kp_input, desc_input = detector.detectAndCompute(input_image, None)
    print('model - %d features, input - %d features' % (len(kp_model), len(kp_input)))

    # --------- STEP 2: FEATURE MATCHING (FLANN OR BRUTE FORCE) AND HOMOGRAPHY  -------------------------
    (mask, p_model_good, p_input_good, H, H2) = feat_ops.match_and_draw('find_obj', matcher, desc_model,
                                                                        desc_input, kp_model, kp_input,
                                                                        model_image, input_image, False)

    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # --------- STEP 3: VALIDATE HOMOGRAPHY/PERSPECTIVE MATRIX ----------------------
    if(feat_ops.validate_homography(H)):  # H = perspective transformation matrix
        print("!!Valid HOMOGRAPHY!!")
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

    '''--------- STEP 3.2: EVENTUEEL OOK REPROJECTION ERROR BEREKENEN ---------------------'''
    # Check the Reprojection error:  https://stackoverflow.com/questions/11053099/how-can-you-tell-if-a-homography-matrix-is-acceptable-or-not
    # https://en.wikipedia.org/wiki/Reprojection_error
    # Calc the euclidean distance for the first 3 features
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #my_model_pts2 = np.float32(model_pts_2D).reshape(-1, 1, 2)  # bit reshapeing so the cv2.perspectiveTransform() works
    #model_transform_pts_2D = cv2.perspectiveTransform(my_model_pts2,
    #                                                  perspective_trans_matrix)  # transform(input_pts_2D)
    #model_transform_pts_2D = np.squeeze(model_transform_pts_2D[:])  # strip 1 dimension
    #
    # TODO: eerst normaliseren
    # TODO: gaan we dit echt doen? volgende validate stappen lijken veel robuster en geen normalisering nodig
    # reprojection_error = (((model_transform_pts_2D[0, 0] - input_pts_2D[0, 0]) ** 2 + (
    # model_transform_pts_2D[0, 1] - input_pts_2D[0, 1]) ** 2)
    #                       + ((model_transform_pts_2D[1, 0] - input_pts_2D[1, 0]) ** 2 + (
    # model_transform_pts_2D[1, 1] - input_pts_2D[1, 1]) ** 2)
    #                       + ((model_transform_pts_2D[2, 0] - input_pts_2D[2, 0]) ** 2 + (
    # model_transform_pts_2D[2, 1] - input_pts_2D[2, 1]) ** 2)) ** 0.5


    '''--------- STEP 3.3 APPEND HUMAN POSE FEATURES ----------------------------------'''
    # append pose features   => GEBEURT NU IN FUNCTIES ZELF
    #p_model_good = np.append(p_model_good, [model_pose_features[0]], 0)
    #p_model_good = np.append(p_model_good, [model_pose_features[1]], 0)

    #p_input_good = np.append(p_input_good, [input_pose_features[0]], 0)
    #p_input_good = np.append(p_input_good, [input_pose_features[1]], 0)

    p_input_good_incl_pose = np.vstack((p_input_good, input_pose_features))
    p_model_good_incl_pose = np.vstack((p_model_good, model_pose_features))

    '''--------- STEP 4: PERSPECTIVE CORRECTION  (eliminate perspective distortion) ------------- '''
    (p_persp_trans_input, input_pose_trans, persp_trans_input_img ) = feat_ops.perspective_correction(H2, p_model_good_incl_pose, p_input_good_incl_pose,
                                                                                    model_pose_features, input_pose_features,
                                                                                    model_image, input_image)


    '''--------- STEP 4.2: ?? Kmeans - CLUSTERING THE FEATURES  ?? -------------  '''


    '''--------- STEP 5: INTERACTION BETWEEN HUMAN AND URBAN SCENE Without perspective correction------------------ '''
    # Calc affine trans between the wrest points and some random feature points of the building
    # The question is: WHICH feature points should we take??
    # An option is to go for the "best matches" (found during featuring-matching)
    # An other option is just to take an certain number of random matches
    # Third option would be to take all the building feature points,
    # but that would probably limit transformation in aspect of the mutual spatial
    # relation between the person and the building
    #feat_ops.affine_trans_interaction_both(p_model_good, p_input_good, model_pose_features, input_pose_features,  model_image, input_image, "both")

    print("\n----------- only_pose without correction -------------")
    feat_ops.affine_trans_interaction_only_pose(p_model_good, p_input_good, model_pose_features, input_pose_features,
                                           model_image, input_image, "only_pose")

    #print("----RAAAAND: ")
    #feat_ops.affine_trans_interaction_pose_rand_scene(p_model_good, p_input_good, model_pose_features, input_pose_features,
    #                                            model_image, input_image, "rand")



    '''--------- STEP 5: INTERACTION BETWEEN HUMAN AND URBAN SCENE WiITH perspective correction------------------ '''
    print("\n----------- only_pose WITH COrREctiOnN-------------")
    p_input_persp_only_buildings = p_persp_trans_input[0:len(p_persp_trans_input) - len(input_pose_features)]

    feat_ops.affine_trans_interaction_only_pose(p_model_good, p_input_persp_only_buildings, model_pose_features, input_pose_trans,
                                                model_image, persp_trans_input_img, "only_pose incl correct")


    # p_input_persp_only_buildings = p_persp_trans_input[0:len(p_persp_trans_input)-len(input_pose_features)]
    # feat_ops.affine_trans_interaction_both(p_model_good, p_input_persp_only_buildings,
    #                                   model_pose_features, input_pose_trans,
    #                                   model_image, persp_trans_input_img, "incl persp corr")


    plt.show()

