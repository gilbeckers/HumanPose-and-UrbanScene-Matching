
import logging
import logging

import dataset.Multipose_dataset_actions as dataset
logger = logging.getLogger(__name__)
path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'

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
dataset.check_matches("00100")
'''
'''
logging.basicConfig(level=logging.DEBUG)
dataset.test_script()
'''
'''
dataset.replace_json_files("1")
'''
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
dataset. draw_pr_curve()
