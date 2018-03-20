import sys

import cv2
from matplotlib import pyplot as plt
from urbanscene.urban_scene import match_scene_multi
from posematching.multi_person import match
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("urban scene matching (multi)-- ")

import common
from urbanscene import features

feature_name = 'sift-flann'
path_img = '../img/'  #'posesGeoteam/fotos/'
path_json = '../json_data/'   #'posesGeoteam/json/'
model_name = 'duo22.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'duo24.jpg'  # goeie : "pisa10"  taj4  # trap2     trap3
model_image = cv2.imread(path_img + model_name, cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread(path_img + input_name, cv2.IMREAD_GRAYSCALE)

detector, matcher = features.init_feature(feature_name)

if model_image is None:
    print('Failed to load fn1:', model_name)
    sys.exit(1)

if input_image is None:
    print('Failed to load fn2:', input_image)
    sys.exit(1)

if detector is None:
    print('unknown feature:', feature_name)
    sys.exit(1)

logger.debug(" using %s", feature_name)


logger.debug("---Starting pose matching --")
model_pose_features = common.parse_JSON_multi_person(path_json + model_name.split('.')[0] +  '.json')  # + '_keypoints'
input_pose_features = common.parse_JSON_multi_person(path_json + input_name.split('.')[0] +  '.json')

result_pose_matching = match(model_pose_features, input_pose_features)

logger.debug("---Result pose matching: --")
logger.debug(result_pose_matching)

if result_pose_matching.match_bool:
    logger.debug(result_pose_matching.matching_permutations)

else:
    logger.debug("No matching poses found, so quit URBAN SCENE MATCHING")
    exit()




logger.debug("--- Starting urbanscene matching ---")
# Loop over all found matching comnbinations
# And order input poses according to matching model poses
for matching_permuations, score in result_pose_matching.matching_permutations.items():
    ordered_model = []
    ordered_input = []

    for model_index, input_index in enumerate(matching_permuations):
        ordered_model.append(model_pose_features[model_index])
        ordered_input.append(input_pose_features[input_index])

    ordered_input = np.vstack(ordered_input)
    ordered_model = np.vstack(ordered_model)
    (result, error) = match_scene_multi(detector, matcher, model_image, input_image, ordered_model,
                      ordered_input)
    logger.info("Match result: %s   score:%f", str(result), round(error,4))


plt.show()