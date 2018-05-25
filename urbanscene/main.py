import sys
import cv2
import common
from urbanscene import features
import logging
import matching
from matplotlib import pyplot as plt
import timer
import numpy as np
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")
import plot_vars


feature_name = 'orb-flann'
#path_img = '../img/galabal2018/fotos/'    #'../img/'  #'posesGeoteam/fotos/'
#path_json = '../img/galabal2018/json/' #'../json_data/'   #'posesGeoteam/json/'


path_img = '../img/kever2/'  # 'posesGeoteam/fotos/'
path_json = '../json_data/kever2/'  # 'posesGeoteam/json/'

# path_img = "../img/"
# path_json = '../json_data/'  # 'posesGeoteam/json/'

model_name = 'kever17.jpg' #'bulb14.jpg'  # goeie : "pisa9"  taj3  # trap1     trap1   'kever197.jpg'
input_name = 'kever70.jpg' #'bulb16.jpg' # goeie : "pisa10"  taj4  # trap2     trap3

#model_name="duo43.jpg"
#input_name="duo48.jpg"

model_image = cv2.imread(path_img + model_name, cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread(path_img + input_name, cv2.IMREAD_GRAYSCALE)


include_keypoints = True
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

plot_us = True  # plot urban scene
plot_mp = False # plot multi pose
plot_vars.input_name = input_name
plot_vars.model_name = model_name
plot_vars.model_path = path_img + model_name
plot_vars.input_path = path_img + input_name
plot_vars.write_img = True
plot_vars.plot_type = "extreme"#"illustrate_crop"


logger.debug("---Starting pose matching --")
if include_keypoints:
    model_pose_features = common.parse_JSON_multi_person(path_json + model_name.split('.')[0] + '_keypoints' +  '.json')  # + '_keypoints'
    input_pose_features = common.parse_JSON_multi_person(path_json + input_name.split('.')[0]  +'_keypoints' + '.json')
else:
    model_pose_features = common.parse_JSON_multi_person(path_json + model_name.split('.')[0] +  '.json')  # + '_keypoints'
    input_pose_features = common.parse_JSON_multi_person(path_json + input_name.split('.')[0]  +'.json')




plot_vars.input_pose_org = np.vstack(input_pose_features)
plot_vars.model_pose_org = np.vstack(model_pose_features)

# make undetected points [0, 0] so they aren't plot in final result-plots
plot_vars.input_pose_org[plot_vars.input_pose_org == [0, 0]] = np.nan
plot_vars.model_pose_org[plot_vars.model_pose_org == [0, 0]] = np.nan



result_whole = matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image,plot_us, plot_mp)

if plot_us or plot_mp:
    plt.show()

