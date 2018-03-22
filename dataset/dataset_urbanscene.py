import json
import logging
import sys
import cv2
import urbanscene.labels as labels
import numpy as np
from matplotlib import pyplot as plt
import common
from urbanscene import features

import matching

feature_name = 'orb-flann'
path_img = '../img/'  # 'posesGeoteam/fotos/'
path_json = '../json_data/'  # 'posesGeoteam/json/'
model_name = 'dart'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'dart'  # goeie : "pisa10"  taj4  # trap2     trap3
img_type = '.jpg'

start_counter = 4
end_counter = 13

thresh = 0.154
thresh_sum = 22
logging.basicConfig(filename='dataset_'+ model_name+'.log',level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


detector, matcher = features.init_feature(feature_name)
if detector is None:
    print('unknown feature:', feature_name)
    sys.exit(1)

logging.info("start iterating over dataset, group = " + model_name)
logging.info('using ' + feature_name  + "  threshold=" + str(thresh) + "  sum_thresh=" + str(thresh_sum) + "(not used)")

plot = True
include_keypoints = True

for i in range(start_counter, end_counter+1):

    model_image = cv2.imread(path_img + model_name + str(i) + img_type, cv2.IMREAD_GRAYSCALE)
    if model_image is None:
        logging.info('Failed to load fn1: %s', model_name + str(i) + img_type)
        continue
        #sys.exit(1)

    #model_pose_features = common.parse_JSON_single_person(path_json + model_name + str(i) + '_keypoints' + '.json')  # + '_keypoints'
    if include_keypoints:
        model_pose_features = common.parse_JSON_multi_person(path_json + model_name + str(i) + '_keypoints' + '.json')  # + '_keypoints'
    else:
        model_pose_features = common.parse_JSON_multi_person(
            path_json + model_name + str(i) + '.json')  # + '_keypoints'

    intermediate_result = {}

    for j in range(start_counter, end_counter+1):
        logging.info("Matching model(%d) with input(%d)", i, j)
        if i == j:
            logging.info("--> Skip: don't compare the same image")
            continue  # don't compare the same image

        input_image = cv2.imread(path_img + input_name + str(j) + img_type, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            logging.info('Failed to load fn2:', input_name + str(j) + img_type)
            #sys.exit(1)
            continue

        if include_keypoints:
            #input_pose_features = common.parse_JSON_single_person(path_json + input_name + str(j) + '_keypoints' + '.json')
            input_pose_features = common.parse_JSON_multi_person(path_json + input_name + str(j) + '_keypoints' + '.json')
        else:
            input_pose_features = common.parse_JSON_multi_person(path_json + input_name + str(j) + '.json')
        copy_model_pose_features = list(model_pose_features)

        result_pose_matching = matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image, plot)
        if plot:
            plt.show()

if plot:
    plt.show()
