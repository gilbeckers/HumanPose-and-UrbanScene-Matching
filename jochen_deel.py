
import logging
import logging

import dataset.Multipose_dataset_actions as dataset
logger = logging.getLogger(__name__)


#**************************pr curves*********************

logging.basicConfig(level=logging.ERROR)
dataset.draw_pr_curve()
