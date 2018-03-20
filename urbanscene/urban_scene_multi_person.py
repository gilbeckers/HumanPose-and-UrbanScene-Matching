import sys

import cv2
from matplotlib import pyplot as plt
from urbanscene.urban_scene_single_person import match_scene_single_person
from posematching.multi_person import match
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("urban scene matching (multi)")

import common
from urbanscene import feat_ops

feature_name = 'sift-flann'
path_img = '../img/'  #'posesGeoteam/fotos/'
path_json = '../json_data/'   #'posesGeoteam/json/'
model_name = 'duo22.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'duo24.jpg'  # goeie : "pisa10"  taj4  # trap2     trap3
model_image = cv2.imread(path_img + model_name, cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread(path_img + input_name, cv2.IMREAD_GRAYSCALE)

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


model_pose_features = common.parse_JSON_multi_person(path_json + model_name.split('.')[0] +  '.json')  # + '_keypoints'
input_pose_features = common.parse_JSON_multi_person(path_json + input_name.split('.')[0] +  '.json')

result_pose_matching = match(model_pose_features, input_pose_features)

if result_pose_matching.match_bool:
    logger.debug(result_pose_matching.matching_permutations)

else:
    logger.debug("No matching poses found, so quit URBAN SCENE MATCHING")
    exit()





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
    match_scene_single_person(detector, matcher, model_image, input_image, ordered_model ,
                              ordered_input)









plt.show()