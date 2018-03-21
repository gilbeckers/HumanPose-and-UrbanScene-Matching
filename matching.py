import sys

import cv2
from matplotlib import pyplot as plt
from urbanscene.urban_scene import match_scene_multi
from posematching.multi_person import match
import numpy as np

import logging
logger = logging.getLogger("-MAIN matching- ")

import common
from urbanscene import features


# Performs the whole matching
# First multi pose matching, followed by urbanscene matching
def match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image):

    result_pose_matching = match(model_pose_features, input_pose_features)
    #logger.debug("---Result pose matching: --")
    #logger.debug(result_pose_matching)

    if result_pose_matching.match_bool:
        #logger.debug(result_pose_matching.matching_permutations)
        logger.debug("Pose matching succes!")

    else:
        logger.debug("No matching poses found, so quit URBAN SCENE MATCHING")
        exit()

    logger.debug("--- Starting urbanscene matching ---")
    # Loop over all found matching comnbinations
    # And order input poses according to matching model poses
    for matching_permuations, result in result_pose_matching.matching_permutations.items():
        model_poses = result['model']
        input_poses = result['input']

        (result, error) = match_scene_multi(detector, matcher,
                                            model_image, input_image,
                                            model_poses,input_poses)
        logger.info("Match result: %s   score:%f", str(result), round(error, 4))

    plt.show()


    return