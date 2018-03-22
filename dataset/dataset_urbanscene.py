
import logging
import sys
import cv2
from matplotlib import pyplot as plt
import common
from urbanscene import features
import matching
import numpy as np

feature_name = 'orb-flann'
path_img = '../img/'  # 'posesGeoteam/fotos/'
path_json = '../json_data/'  # 'posesGeoteam/json/'
model_name = 'duo'  # goeie : "pisa9"  taj3  # trap1     trap1
input_name = 'duo'  # goeie : "pisa10"  taj4  # trap2     trap3
img_type = '.jpg'

start_counter = 28
end_counter = 51

# logging.basicConfig(filename='dataset_'+ model_name+'.log',level=logging.INFO)
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# rootLogger = logging.getLogger()
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MAIN")


detector, matcher = features.init_feature(feature_name)
if detector is None:
    print('unknown feature:', feature_name)
    sys.exit(1)

logger.info("start iterating over dataset, group = " + model_name)
logger.info('using ' + feature_name )

plot = True
include_keypoints = False

for i in range(start_counter, end_counter+1):

    model_image = cv2.imread(path_img + model_name + str(i) + img_type, cv2.IMREAD_GRAYSCALE)
    if model_image is None:
        logger.info('Failed to load fn1: %s', model_name + str(i) + img_type)
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
        logger.info("Matching model(%d) with input(%d)", i, j)
        if i == j:
            logger.info("--> Skip: don't compare the same image")
            continue  # don't compare the same image

        input_image = cv2.imread(path_img + input_name + str(j) + img_type, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            logger.info('Failed to load fn2:', input_name + str(j) + img_type)
            #sys.exit(1)
            continue

        if include_keypoints:
            #input_pose_features = common.parse_JSON_single_person(path_json + input_name + str(j) + '_keypoints' + '.json')
            input_pose_features = common.parse_JSON_multi_person(path_json + input_name + str(j) + '_keypoints' + '.json')
        else:
            input_pose_features = common.parse_JSON_multi_person(path_json + input_name + str(j) + '.json')
        copy_model_pose_features = list(model_pose_features)

        (result_pose_matching, result_whole) = matching.match_whole(model_pose_features, input_pose_features,detector, matcher,model_image, input_image, plot=False)
        result_string = ""

        if result_whole:
            result_string=" :) MATCH :)"
        elif result_pose_matching:
            result_string=" :( NO MATCH :( (poses match anyway)"
        else:
            result_string=" :( NO MATCH! :("
        if plot:
            plt.show()
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
            implot = ax1.imshow(np.asarray(model_image), cmap='gray')
            # ax1.set_title(model_image_name + ' (model)')
            ax1.set_title("model "+ model_name + str(i)+ result_string)

            # ax2.set_title(input_image_name + ' (input)')
            ax2.set_title("input "+ input_name+ str(j)+ result_string)
            ax2.imshow(np.asarray(input_image), cmap='gray')
            plt.show()



if plot:
    plt.show()
