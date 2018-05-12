

from posematching.multi_person import match
from posematching.multi_person import plot_poses
import matplotlib.pyplot as plt
import numpy as np
from common import parse_JSON_multi_person
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_gil")
json_data_path = '../img/galabal2018/json/' #'../json_data/'
images_data_path = '../img/galabal2018/fotos/' #'../img/'

json_data_path = '../json_data/' #'../json_data/'
images_data_path = '../img/abbey/' #'../img/'

json_data_path = '../specials/MP_overlapping/' #'../json_data/'
images_data_path = '../specials/MP_overlapping/' #'../img/'

#json_data_path = '../json_data/' #'../json_data/'
#images_data_path = '../img/' #'../img/'


'''
-------------------- MULTI PERSON -------------------------------------
'''

model = "09958"# "duo3"  #  "09958" #
input = "08425"  #"duo4"  # "08425"  #
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'
model_features = parse_JSON_multi_person(model_json)
input_features = parse_JSON_multi_person(input_json)


plot_poses(np.vstack(model_features), np.vstack(input_features), model_image_name=model_image, input_image_name=input_image)
matchresult = match(model_features, input_features, normalise=True, plot=True, model_image=model_image, input_image=input_image)
#logger.info("Match result: %s", str(matchresult.match_bool))
plt.show()


