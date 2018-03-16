import posematching.z_multiperson_match as jochen


from common import parse_JSON_multi_person, parse_JSON_multi_person_jochen
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match_jochen")
json_data_path = '../json_data/'
images_data_path = '../img/'

'''
-------------------- MULTI PERSON -------------------------------------
'''

model = "duo22"
input = "duo24"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'
model_features = parse_JSON_multi_person_jochen(model_json)
input_features = parse_JSON_multi_person_jochen(input_json)

#jochen.find_best_match(model_features, input_features)

(result, error_score, input_transform) = jochen.multi_person_ordered(model_features, input_features, True)
print(result)

#jochen.multi_person(model_features, input_features)


