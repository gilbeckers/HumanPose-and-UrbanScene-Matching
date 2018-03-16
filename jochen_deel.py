import posematching.z_multiperson_match as jochen
import dataset.Multipose_dataset_actions as dataset

from common import parse_JSON_multi_person, parse_JSON_multi_person_jochen
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match_jochen")
json_data_path = '../json_data/'
images_data_path = '../img/'

data = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/json/'
galabal = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/poses/'
galabaljson = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/json/'
galabalfotos = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/fotos/'
'''
-------------------- MULTI PERSON -------------------------------------
'''
logging.basicConfig(level=logging.DEBUG)
dataset.check_matches("00100")
#jochen.multi_person(model_features, input_features)
