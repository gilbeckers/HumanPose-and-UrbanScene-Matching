import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import numpy as np
from common import feature_scaling, handle_undetected_points, \
    split_in_face_legs_torso, find_transformation, unsplit, split_in_face_legs_torso_v2
from dataset import Multipose_dataset_actions as dataset #eucl_dis_tresh_torso, rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh
import thresholds
import posematching.pose_comparison as pose_comparison
logger = logging.getLogger("single_person")

# Init the returned tuple
MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])
MatchResultMulti = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation", "matching_permutations"])

class MatchCombo(object):
    def __init__(self, error_score, input_id, model_id, model_features, input_features, input_transformation):
        self.error_score = error_score
        self.input_id = input_id
        self.model_id = model_id
        self.model_features = model_features # niet noodzaakelijk voor logica, wordt gebruikt voor plotjes
        self.input_features = input_features # same
        self.input_transformation = input_transformation

'''------------------------------------------------------------------------'''
'''-----------------SINGLE PERSOON ----------------------'''
'''---------------------------------------------------------------'''
'''
Description single_person(model_features, input_features):
GOAL: Decides if the inputpose matches with the modelpose.

Both valid and unvalid modelposes are allowed, that is, modelposes with no undetected body parts are also allowed.
If a unvalid model pose is used, the inputpose is adjusted and the matching continues.

The inputpose is also allowed to contain a number of undetected body-parts.
These undetected features are marked as (0,0). The algorithm is designed to handle these incomplete situations as follow:

In order to proceed the matching with these undetected points, a copy is made of the model pose
where the corresponding undetected points of the input pose are also set to (0,0).
-- So, the inputpose and modelpose_copy still have the same length of features (18) and also the same
amount of undetected features. --

Later, before the affine transformation is found, the undetected features are temporarily filtered out.
In this way these origin points don't influence the least-squares algorithm.



In case of undetected features in the inputpose, one should care of the following:
NOTE 1: The (0,0) points can't just be deleted because
without them the feature-arrays would become ambigu. (the correspondence between model and input)

NOTE 2: In order to disregard the undetected feauters of the inputpose, the corresponding modelpose features
are also altered to (0,0). Because we don't want the loose the original information of the complete modelpose,
first a local copy is made of the modelpose before the altering. The rest of the function (the actual matching)
uses this copy. At the end, the original (the unaltered version) model is returned, so the next step in the pipeline
still has all the original data.

NOTE 3: the acceptation and introduction of (0,0) points is a danger for our current normalisation
These particular origin points should not influence the normalisation
(which they do if we neglect them, xmin and ymin you know ... )



Parameters:
Takes two parameters, model name and input name.
Both have a .json file in json_data and a .jpg or .png in image_data
@:param model_features:
@:param input_features:
@:param normalise:

Returns:
@:returns result matching
@:returns error_score
@:returns input_transformation
@:returns model_features => is needed in multi_person2() and when (0,0) are added to modelpose
'''


def match_single(model_features, input_features, normalise=True):
    # Filter the undetected features and mirror them in the other pose
    (input_features_copy, model_features_copy) = handle_undetected_points(input_features, model_features)

    non_zero_rows = np.count_nonzero((input_features_copy != 0).sum(1))
    zero_rows = len(input_features_copy) - non_zero_rows
    if zero_rows > 4:
        logger.debug("Model has more feature then input therefore not matched")
        result = MatchResult(False,
                             error_score=0,
                             input_transformation=None)
        #return result

    assert len(model_features_copy) == len(input_features_copy)

    if (normalise):
        model_features_copy = feature_scaling(model_features_copy)
        input_features_copy = feature_scaling(input_features_copy)

    # Split features in three parts
    (model_face, model_torso, model_legs) = split_in_face_legs_torso_v2(model_features_copy)
    (input_face, input_torso, input_legs) = split_in_face_legs_torso_v2(input_features_copy)

    # In case of no normalisation, return here (ex; plotting)
    # Without normalisation the thresholds don't say anything
    #   -> so comparison is useless
    if (not normalise):
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=None)
        return result

    ######### THE THRESHOLDS #######
    eucl_dis_tresh_face = thresholds.SP_DISTANCE_FACE
    eucl_dis_tresh_torso = thresholds.SP_DISTANCE_TORSO
    rotation_tresh_torso = thresholds.SP_ROTATION_TORSO
    eucl_dis_tresh_legs = thresholds.SP_DISTANCE_LEGS
    rotation_tresh_legs = thresholds.SP_ROTATION_LEGS
    eucld_dis_shoulders_tresh =thresholds.SP_DISTANCE_SHOULDER




    ################################


    # TODO @j3 keer het zelfde!! -> bad code design :'(
    (input_transformed_face, transformation_matrix_face) = find_transformation(model_face, input_face)
    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    if np.count_nonzero(model_face) > 8:
        if (np.count_nonzero(model_face) - np.count_nonzero(input_face)) < 2:
            #
            #
            result_face = True
        else:
            logger.debug("Model has more face feature then input therefore not matched %d" , (np.count_nonzero(model_face) - np.count_nonzero(input_face)) )
            result_face = False
    else:
        logger.debug("too less points for face in model so face match")
        result_face = True

    rotation_torso=0
    (input_transformed_torso, transformation_matrix_torso) = find_transformation(model_torso, input_torso)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso,
                                                                                     input_transformed_torso)
    if (np.count_nonzero(model_torso) > 4):
        if (np.count_nonzero(model_torso) - np.count_nonzero(input_torso)) < 2:

            (result_torso,rotation_torso) = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso,
                                                                       transformation_matrix_torso,
                                                                       eucl_dis_tresh_torso, rotation_tresh_torso,
                                                                       max_euclidean_error_shoulders,
                                                                       eucld_dis_shoulders_tresh)
        else:
            logger.debug("Model has more Torso feature then input therefore not matched %d", (np.count_nonzero(model_torso) - np.count_nonzero(input_torso)))
            result_torso = False
    else:
        logger.debug("too less points for Torso in model so Torso match %d",np.count_nonzero(model_torso)  )
        result_torso = True

    # handle legs
    rotation_legs =0
    (input_transformed_legs, transformation_matrix_legs) = find_transformation(model_legs, input_legs)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)
    if (np.count_nonzero(model_legs) > 8):
        if (np.count_nonzero(model_legs) - np.count_nonzero(input_legs)) < 2:
            (result_legs,rotation_legs) = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,eucl_dis_tresh_legs, rotation_tresh_legs)
            logger.debug("Model legs zeros: %d",np.count_nonzero(model_legs))
        else:
            logger.debug("Model has more legs feature then input therefore not matched %d", (np.count_nonzero(model_legs) - np.count_nonzero(input_legs)) )
            result_legs = False
    else:
        logger.debug("too less points for legs in model so legs match %d", np.count_nonzero(model_legs))
        result_legs = True

    # Wrapped the transformed input in one whole pose
    input_transformation = unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)

    # TODO: construct a solid score algorithm
    error_score = ((max_euclidean_error_torso/eucl_dis_tresh_torso) + (max_euclidean_error_legs/eucl_dis_tresh_legs) + (max_euclidean_error_shoulders/eucld_dis_shoulders_tresh)+(rotation_legs/rotation_tresh_legs)+(rotation_torso/rotation_tresh_torso))/5

    result = MatchResult((result_torso and result_legs and result_face),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result


#Plot the calculated transformation on the model image
#And some other usefull plots for debugging
#NO NORMALIZING IS DONE HERE BECAUSE POINTS ARE PLOTTED ON THE ORIGINAL PICTURES!
def plot(model_features, input_features, model_image_name, input_image_name, input_title = "input",  model_title="model",
                       transformation_title="transformation of input + model"):

    # Filter the undetected features and mirror them in the other pose
    (input_features_copy, model_features_copy) = handle_undetected_points(input_features, model_features)

    # plot vars
    markersize = 3

    #Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    # Split features in three parts
    (model_face, model_torso, model_legs) = split_in_face_legs_torso(model_features_copy)
    (input_face, input_torso, input_legs) = split_in_face_legs_torso(input_features_copy)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = find_transformation(model_face,
                                                                                                     input_face)
    (input_transformed_torso, transformation_matrix_torso) = find_transformation(model_torso,
                                                                                                       input_torso)
    (input_transformed_legs, transformation_matrix_legs) = find_transformation(model_legs,
                                                                                                     input_legs)


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    #ax1.set_title(model_image_name + ' (model)')
    ax1.set_title(model_title)
    ax1.plot(*zip(*model_features_copy), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    #ax2.set_title(input_image_name + ' (input)')
    ax2.set_title(input_title)
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features_copy), marker='o', color='r', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    whole_input_transform = unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)
    ax3.set_title(transformation_title)
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features_copy), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*whole_input_transform), marker='o', color='b', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'), mpatches.Patch(color='magenta', label='model')])

    plt.show(block=False)
