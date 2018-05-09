
import logging
import logging

import dataset.Multipose_dataset_actions as dataset
import posematching.multi_person as multiperson
import posematching.calcAngle as calcAngle
import common
logger = logging.getLogger(__name__)
path = '/media/jochen/2FCA69D53AB1BFF43/dataset/poses/'
testdata = '/media/jochen/2FCA69D53AB1BFF43/dataset/specials/'

'''
logging.basicConfig(level=logging.INFO)
for i in range(1,10):
    print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    print i
    print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    Multipose_dataset_actions.find_matches_with("0000"+str(i))
'''

'''
logging.basicConfig(level=logging.ERROR)
dataset.find_matches_with("00100")
'''
'''
logging.basicConfig(level=logging.CRITICAL)
dataset.check_matches("1")
'''
'''
logging.basicConfig(level=logging.DEBUG)
dataset.test_script()
'''
'''
dataset.replace_json_files("1")
'''

#dataset.finderror()

#*********************galabal*********************
'''
logging.basicConfig(level=logging.DEBUG)
dataset.test_script()
'''
'''
logging.basicConfig(level=logging.INFO)
dataset.find_galabal_matches("1")
'''
'''
logging.basicConfig(level=logging.DEBUG)
dataset.check_galabal_matches("4")
'''

#**************************pr curves*********************


logging.basicConfig(level=logging.ERROR)
dataset.draw_pr_curve()

# logging.basicConfig(level=logging.ERROR)
# dataset.findSpecials()

# #******************-*************quick tests******************
#
# logging.basicConfig(level=logging.ERROR)
# model =  '/media/jochen/2FCA69D53AB1BFF43/dataset/poses/pose1/json/0.json'
# input = '/media/jochen/2FCA69D53AB1BFF43/dataset/poses/pose2/json/5.json'
# model_features = common.parse_JSON_multi_person(model)
# input_features = common.parse_JSON_multi_person(input)
# primary_angles = calcAngle.prepareangles(model_features)
# print(primary_angles)
# secondary_angles = calcAngle.prepareangles(input_features)
# print(secondary_angles)
# result, angles = calcAngle.succes(primary_angles, secondary_angles,15)
# print(result)
# print(angles)
