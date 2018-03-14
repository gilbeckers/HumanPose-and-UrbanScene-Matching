import cv2
import feat_ops
import numpy as np
import sys
import logging


FLANN_INDEX_KDTREE  = 1  # bug: flann enums are missing
FLANN_INDEX_LSH     = 6
MIN_MATCH_COUNT     = 4

def match_scene_single_person(detector, matcher, model_image, input_image,model_pose_features, input_pose_features, thresh):
    assert model_pose_features.shape == input_pose_features.shape
    ''' ---------- STEP 1: FEATURE DETECTION AND DESCRIPTION (ORB, SIFT, SURF, BRIEF, ASIFT -------------------- '''
    kp_model, desc_model = detector.detectAndCompute(model_image, None)
    kp_input, desc_input = detector.detectAndCompute(input_image, None)
    #print('model - %d features, input - %d features' % (len(kp_model), len(kp_input)))
    logging.debug('model - %d features, input - %d features' % (len(kp_model), len(kp_input)))

    ''' --------- STEP 2: FEATURE MATCHING (FLANN OR BRUTE FORCE) AND HOMOGRAPHY  ------------------------- '''
    (mask, p_model_good, p_input_good, H, H2) = feat_ops.match_and_draw("multipips", matcher, desc_model,
                                                                        desc_input, kp_model, kp_input,
                                                                        model_image, input_image, False)

    cv2.waitKey()
    cv2.destroyAllWindows()

    ''' --------- STEP 3: VALIDATE HOMOGRAPHY/PERSPECTIVE MATRIX ---------------------- '''
    if (feat_ops.validate_homography(H)):  # H = perspective transformation matrix
        logging.debug("!!Valid HOMOGRAPHY!!")
        # else:
        # exit()

    else:
        logging.debug("UNVALID HOMOGRAPHY")
        return (1,1,1000,1000)
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

    '''--------- STEP 3.1 APPEND HUMAN POSE FEATURES ----------------------------------'''
    # append pose features   => GEBEURT NU IN FUNCTIES ZELF
    p_input_good_incl_pose = np.vstack((p_input_good, input_pose_features))
    p_model_good_incl_pose = np.vstack((p_model_good, model_pose_features))

    '''--------- STEP 4: PERSPECTIVE CORRECTION  (eliminate perspective distortion) ------------- '''
    (p_persp_trans_input, input_pose_trans, persp_trans_input_img) = feat_ops.perspective_correction(H2,
                                                                                                     p_model_good_incl_pose,
                                                                                                     p_input_good_incl_pose,
                                                                                                     model_pose_features,
                                                                                                     input_pose_features,
                                                                                                     model_image,
                                                                                                     input_image,
                                                                                                     False)

    '''--------- STEP 5: INTERACTION BETWEEN HUMAN AND URBAN SCENE Without perspective correction------------------ '''
    # Calc affine trans between the wrest points and some random feature points of the building
    # The question is: WHICH feature points should we take??
    # An option is to go for the "best matches" (found during featuring-matching)
    # An other option is just to take an certain number of random matches
    # Third option would be to take all the building feature points,
    # but that would probably limit transformation in aspect of the mutual spatial
    # relation between the person and the building
    # feat_ops.affine_trans_interaction_both(p_model_good, p_input_good, model_pose_features, input_pose_features,  model_image, input_image, "both")

    '''--------- STEP 5: INTERACTION BETWEEN HUMAN AND URBAN SCENE WiITH perspective correction------------------ '''
    logging.debug("\n----------- both WITH COrREctiOnN & SOME RanDOm FeaTuREs-------------")
    p_input_persp_only_buildings = p_persp_trans_input[0:len(p_persp_trans_input) - len(input_pose_features)]

    (err_torso, err_legs, sum_err_torso, sum_err_legs) = feat_ops.affine_trans_interaction_pose_rand_scene(p_model_good, p_input_persp_only_buildings, model_pose_features,
                                                      input_pose_trans,
                                                      model_image, persp_trans_input_img, "pose + random scene", False)



    #plt.show()


    return (err_torso, err_legs, sum_err_torso, sum_err_legs)