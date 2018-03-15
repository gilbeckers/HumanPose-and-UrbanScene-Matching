import posematching.z_multiperson_match as jochen


from common import parse_JSON_multi_person, parse_JSON_multi_person_jochen
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match_jochen")
json_data_path = '../json_data/'
images_data_path = '../img/'
galabal = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/poses/'
galabaljson = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/json/'
galabalfotos = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/fotos/'
'''
-------------------- MULTI PERSON -------------------------------------
'''

pose = "4"
model = galabal+pose+"/json/"+pose+".json"
input = galabal+pose+"/json/148.json"


model_features = parse_JSON_multi_person_jochen(model)
input_features = parse_JSON_multi_person_jochen(input)

jochen.find_best_match(model_features, input_features)

(result, error_score, input_transform) = jochen.multi_person_ordered(model_features, input_features, True)

#jochen.multi_person(model_features, input_features)
