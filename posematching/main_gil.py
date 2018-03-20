#import posematching.rust.z_multiperson_match as jochen

from posematching.multi_person import match

from common import parse_JSON_multi_person, parse_JSON_multi_person_jochen
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_gil")
json_data_path = '../json_data/'
images_data_path = '../img/'

'''
-------------------- MULTI PERSON -------------------------------------
'''

model = "duo3"
input = "duo4"
model_json = json_data_path + model + '.json'
input_json = json_data_path + input + '.json'
model_image = images_data_path + model + '.jpg'
input_image = images_data_path + input + '.jpg'
model_features = parse_JSON_multi_person_jochen(model_json)
input_features = parse_JSON_multi_person_jochen(input_json)

#jochen.find_best_match(model_features, input_features)

matchresult = match(model_features, input_features, True)

logger.info("Match result: %s", str(matchresult.match_bool))


#jochen.multi_person(model_features, input_features)


