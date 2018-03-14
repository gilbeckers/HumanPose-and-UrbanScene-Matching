import cv2
import feat_ops
import numpy as np
from matplotlib import pyplot as plt
from parse_openpose_json import parse_JSON_single_person
from urban_scene_single_person import match_scene_single_person
import sys
import json

feature_name = 'sift-flann'
path_img = 'img/'  # 'posesGeoteam/fotos/'
path_json = 'json_data/'  # 'posesGeoteam/json/'
model_name = 'dart'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'dart'  # goeie : "pisa10"  taj4  # trap2     trap3
img_type = '.jpg'
amount_img = 13 + 1

detector, matcher = feat_ops.init_feature(feature_name)
if detector is None:
    print('unknown feature:', feature_name)
    sys.exit(1)
print('using', feature_name)

plot = False

dataset_result = {}

for i in range(1, amount_img):
    model_image = cv2.imread(path_img + model_name + str(i) + img_type, cv2.IMREAD_GRAYSCALE)
    if model_image is None:
        print('Failed to load fn1:', model_name + str(i) + img_type)
        sys.exit(1)

    model_pose_features = parse_JSON_single_person(path_json + model_name + str(i) + '_keypoints' + '.json')  # + '_keypoints'

    intermediate_result = {}

    for j in range(1, amount_img):
        if i == j:
            continue  # don't compare the same image

        input_image = cv2.imread(path_img + input_name + str(j) + img_type, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            print('Failed to load fn2:', input_name + str(j) + img_type)
            sys.exit(1)

        input_pose_features = parse_JSON_single_person(path_json + input_name + str(j) + '_keypoints' + '.json')
        (result, err_torso, err_legs) = match_scene_single_person(detector, matcher, model_image, input_image, model_pose_features,
                                  input_pose_features)

        intermediate_result[input_name+str(j)] = result

        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
            ax1.imshow(np.asarray(model_image), cmap='gray')
            # ax1.set_title(model_image_name + ' (model)')
            ax1.set_title("M "+ model_name + str(i) + " ->result:" + str(result)  )

            # ax2.set_title(input_image_name + ' (input)')
            ax2.set_title("I " + input_name + str(j) + " torso: " + str(round(err_torso, 3)) + "  legs: " + str(round(err_legs, 3)) )
            ax2.imshow(np.asarray(input_image), cmap='gray')
            plt.show(block=False)

    dataset_result[model_name+str(i)] = intermediate_result


with open('dataset.json', 'w') as json_file:
    json.dump(dataset_result, json_file)


if plot:
    plt.show()
