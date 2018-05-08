#import posematching.rust.z_multiperson_match as jochen

from posematching.multi_person import match
import matplotlib.pyplot as plt
from common import parse_JSON_multi_person
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_gil")
json_data_path = '../img/galabal2018/json/' #'../json_data/'
images_data_path = '../img/galabal2018/fotos/' #'../img/'

json_data_path = '../json_data/' #'../json_data/'
images_data_path = '../img/abbey/' #'../img/'
'''
-------------------- MULTI PERSON -------------------------------------
'''

model = "abbey66" #"duo3"
input = "abbey5"  #"duo4"
model_json = json_data_path + model + '_keypoints.json'
input_json = json_data_path + input + '_keypoints.json'
model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'
model_features = parse_JSON_multi_person(model_json)
input_features = parse_JSON_multi_person(input_json)

#jochen.find_best_match(model_features, input_features)

matchresult = match(model_features, input_features, normalise=True, plot=True, model_image=model_image, input_image=input_image)

logger.info("Match result: %s", str(matchresult.match_bool))
plt.show()

#jochen.multi_person(model_features, input_features)


