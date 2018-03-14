import sys

import cv2
from matplotlib import pyplot as plt

import feat_ops
import common
from urban_scene_single_person import match_scene_single_person

feature_name = 'sift-flann'
path_img = 'img/'  #'posesGeoteam/fotos/'
path_json = 'json_data/'   #'posesGeoteam/json/'
model_name = 'duo3.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'duo1.jpg'  # goeie : "pisa10"  taj4  # trap2     trap3
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


for i in range(0, len(model_pose_features)):
    match_scene_single_person(detector, matcher, model_image, input_image, model_pose_features[i], input_pose_features[i])


plt.show()