from urbanscene.urban_scene import match_scene_multi
import posematching.multi_person as multi_person
from handlers import json
from handlers import features
import logging
import thresholds
import plot_vars
from matplotlib import pyplot as plt
import cv2
import numpy as np
logger = logging.getLogger("match_whole")

import time


def match(model_json, input_json, model_image_path, input_image_path, plot_us=False, plot_mp=False):
    feature_name = 'orb-flann'
    detector, matcher = features.init_feature(feature_name)

    model_pose_features = json.parse_JSON_multi_person(model_json)
    input_pose_features= json.parse_JSON_multi_person(input_json)
    model_image = cv2.imread(model_image_path, cv2.IMREAD_GRAYSCALE)
    print (model_image)
    if model_image == None:
        model_image_path = model_image_path.rsplit('.')[0] +".png"
        print(model_image_path)
        model_image = cv2.imread(model_image_path, cv2.IMREAD_GRAYSCALE)

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    (result, score, input_transform,permutations) = multi_person.match(model_pose_features, input_pose_features)
    logger.debug("--- Starting urbanscene matching ---")
    # Loop over all found matching comnbinations
    # And order input poses according to matching model poses
    min_error = np.inf
    SP_min_error = []
    MP_min_error = 0.0
    US_min_error = 0.0
    if permutations == None:
        return (0,0,0,0)
    for result in permutations:

        model_poses = result['model']
        input_poses = result['input']
        MP_error_score = result['score']
        #logger.debug(model_poses)
        #logger.debug(input_poses)

        model_image_copy = model_image  #TODO make hard copy ??
        input_image_copy = input_image

        US_error_score = match_scene_multi(detector, matcher,
                                            model_image_copy, input_image_copy,
                                            model_poses,input_poses,
                                            plot_us)
        if US_error_score == np.inf:
            US_error_score = 100.0
        fin_score = (US_error_score/thresholds.AFFINE_TRANS_WHOLE_DISTANCE) #((US_error_score/thresholds.AFFINE_TRANS_WHOLE_DISTANCE) + (result['score']))/2
        if  fin_score < min_error: #(error+result['score'])/2 < min_error:
            logger.debug("new min error %0.4f",fin_score)
            #min_error =(error+result['score'])/2

            min_error = fin_score
            SP_min_error = []
            if isinstance(result['single_pose_scores'], tuple)==1:
                SP_min_error = result['single_pose_scores']
            else:
                SP_min_error.append(result['single_pose_scores'])
            MP_min_error = MP_error_score
            US_min_error = US_error_score
        # if error <= thresholds.AFFINE_TRANS_WHOLE_DISTANCE:
        #     logger.info("===> MATCH! permutation %s  score:%0.4f (thresh ca %0.4f)",
        #                 matching_permuations, round(error, 4), thresholds.AFFINE_TRANS_WHOLE_DISTANCE)
        #     return (True,True)
        # else:
        #     logger.info("===> NO-MATCH! permutation %s  score:%0.4f (thresh ca %0.4f)",
        #                 matching_permuations, round(error, 4), thresholds.AFFINE_TRANS_WHOLE_DISTANCE)
        #     return (True,False)
    #plt.show()

    return (make_percentage_score(min_error),US_min_error,MP_min_error,SP_min_error)

def make_percentage_score(error):
    if error == np.inf:
        return 0
    error = error - 0.7
    if error < 0:
        return 100
    elif  (1 - error)*100 > 0:
        return (1 - error)*100
    else:
        return 0
