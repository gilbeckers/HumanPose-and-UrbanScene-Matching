
import logging
import logging

import dataset.Multipose_dataset_actions as dataset
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.ERROR)
# dataset.find_matches_with("17")
#**************************pr curves*********************

logging.basicConfig(level=logging.ERROR)
dataset.draw_pr_curve()
