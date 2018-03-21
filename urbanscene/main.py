import sys
import cv2
import common
from urbanscene import features
import logging
import matching
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("urban scene matching (multi)-- ")


feature_name = 'orb-flann'
path_img = '../img/'  #'posesGeoteam/fotos/'
path_json = '../json_data/'   #'posesGeoteam/json/'
model_name = 'duo21.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'duo25.jpg'  # goeie : "pisa10"  taj4  # trap2     trap3
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

matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image)