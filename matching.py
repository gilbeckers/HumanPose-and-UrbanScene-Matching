from urbanscene.urban_scene import match_scene_multi
import posematching.multi_person as multi_person
import logging
import thresholds
logger = logging.getLogger("match_whole")

import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        logger.critical('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

# Performs the whole matching
# First multi pose matching, followed by urbanscene matching
@timing
def match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image, plot=False):

    result_pose_matching = multi_person.match(model_pose_features, input_pose_features, plot=plot, input_image = input_image, model_image=model_image)
    #logger.debug("---Result pose matching: --")
    #logger.debug(result_pose_matching)

    if result_pose_matching.match_bool:
        #logger.debug(result_pose_matching.matching_permutations)
        logger.info("Pose matching succes!")

    else:
        logger.info("No matching poses found, so quit URBAN SCENE MATCHING")
        return (False,False)
        #exit()

    logger.debug("--- Starting urbanscene matching ---")
    # Loop over all found matching comnbinations
    # And order input poses according to matching model poses
    for matching_permuations, result in result_pose_matching.matching_permutations.items():
        model_poses = result['model']
        input_poses = result['input']
        #logger.debug(model_poses)
        #logger.debug(input_poses)

        model_image_copy = model_image  #TODO make hard copy ??
        input_image_copy = input_image

        error = match_scene_multi(detector, matcher,
                                            model_image_copy, input_image_copy,
                                            model_poses,input_poses,
                                            plot)
        if error <= thresholds.AFFINE_TRANS_WHOLE_DISTANCE:
            logger.info("===> MATCH! permutation %s  score:%0.4f (thresh ca %0.3f)",
                        matching_permuations, round(error, 4), 0.10)
            return (True,True)
        else:
            logger.info("===> NO-MATCH! permutation %s  score:%0.4f (thresh ca %0.3f)",
                        matching_permuations, round(error, 4), 0.10)
            return (True,False)