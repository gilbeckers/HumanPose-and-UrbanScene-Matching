import json
import logging
import sys

import cv2
import labels
import numpy as np
from matplotlib import pyplot as plt
from urban_scene_single_person import match_scene_single_person

import common
from urbanscene import feat_ops

feature_name = 'orb-flann'
path_img = 'img/'  # 'posesGeoteam/fotos/'
path_json = 'json_data/'  # 'posesGeoteam/json/'
model_name = 'dart'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'dart'  # goeie : "pisa10"  taj4  # trap2     trap3
img_type = '.jpg'
amount_img = 13 + 1

thresh = 0.154
thresh_sum = 22
logging.basicConfig(filename='dataset_'+ model_name+'.log',level=logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)





detector, matcher = feat_ops.init_feature(feature_name)
if detector is None:
    print('unknown feature:', feature_name)
    sys.exit(1)

logging.info("start iterating over dataset, group = " + model_name)
logging.info('using ' + feature_name  + "  threshold=" + str(thresh) + "  sum_thresh=" + str(thresh_sum) + "(not used)")

plot = False

dataset_result = {}

for i in range(1, amount_img):
    model_image = cv2.imread(path_img + model_name + str(i) + img_type, cv2.IMREAD_GRAYSCALE)
    if model_image is None:
        print('Failed to load fn1:', model_name + str(i) + img_type)
        continue
        #sys.exit(1)

    model_pose_features = common.parse_JSON_single_person(path_json + model_name + str(i) + '_keypoints' + '.json')  # + '_keypoints'

    intermediate_result = {}

    for j in range(1, amount_img):
        if i == j:
            continue  # don't compare the same image

        input_image = cv2.imread(path_img + input_name + str(j) + img_type, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            print('Failed to load fn2:', input_name + str(j) + img_type)
            #sys.exit(1)
            continue

        input_pose_features = common.parse_JSON_single_person(path_json + input_name + str(j) + '_keypoints' + '.json')
        (err_torso, err_legs, sum_err_torso, sum_err_legs) = match_scene_single_person(detector, matcher, model_image, input_image, model_pose_features,
                                  input_pose_features, thresh)

        label = labels.chech_same_class(model_name, i, j)

        if err_torso < thresh and err_legs < thresh: #and sum_err_legs < thresh_sum and sum_err_torso < thresh_sum:
            logging.debug("#### MATCH!!!  ###")
            match = True
        else:
            logging.debug("#### NO MATCH!! ###")
            match = False

        if label == match:
            intermediate_result[input_name + str(j)] = [match, "ok", round(err_torso, 3), round(err_legs, 3), round(sum_err_torso, 3)]
        else:
            intermediate_result[input_name + str(j)] = [match, "wrong", round(err_torso, 3), round(err_legs, 3), round(sum_err_torso, 3)]
            logging.error("False match! |  model:" + str(i) + " , input:" + str(j)
                          + "  |  false-result: " + str(match)+ " torso %f  legs:%f  sum:%f" , round(err_torso, 3), round(err_legs, 3), round(sum_err_torso, 3))




        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
            ax1.imshow(np.asarray(model_image), cmap='gray')
            # ax1.set_title(model_image_name + ' (model)')
            ax1.set_title("M "+ model_name + str(i) + " ->result:" + str(match)  )

            # ax2.set_title(input_image_name + ' (input)')
            ax2.set_title("I " + input_name + str(j) + " torso: " + str(round(err_torso, 3)) + "  legs: " + str(round(err_legs, 3)) )
            ax2.imshow(np.asarray(input_image), cmap='gray')
            plt.show(block=False)

    dataset_result[model_name+str(i)] = intermediate_result


with open('dataset.json', 'w') as json_file:
    json.dump(dataset_result, json_file)


if plot:
    plt.show()
