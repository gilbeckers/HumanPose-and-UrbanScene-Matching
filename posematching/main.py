import logging

from dataset import Multipose_dataset_actions

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
logging.basicConfig(level=logging.INFO)
Multipose_dataset_actions.find_matches_with("00100")
'''


'''
logging.basicConfig(level=logging.DEBUG)
Multipose_dataset_actions.check_matches("00001")
'''


#*********************galabal*********************


logging.basicConfig(level=logging.DEBUG)
Multipose_dataset_actions.test_script()

'''
logging.basicConfig(level=logging.INFO)
Multipose_dataset_actions.find_galabal_matches("1")
'''
'''
logging.basicConfig(level=logging.DEBUG)
Multipose_dataset_actions.check_galabal_matches("4")
'''