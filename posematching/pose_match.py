import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import numpy as np
import posematching.proc_do_it as proc_do_it
from common import feature_scaling, handle_undetected_points, \
    split_in_face_legs_torso, find_transformation, unsplit

import posematching.pose_comparison as pose_comparison
logger = logging.getLogger("pose_match")

# Init the returned tuple
MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])

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
def single_person(model_features, input_features, normalise=True):

    # Filter the undetected features and mirror them in the other pose
    (input_features_copy, model_features_copy) = handle_undetected_points(input_features, model_features)

    if (normalise):
        model_features_copy = feature_scaling(model_features_copy)
        input_features_copy = feature_scaling(input_features_copy)

    #Split features in three parts
    (model_face, model_torso, model_legs) = split_in_face_legs_torso(model_features_copy)
    (input_face, input_torso, input_legs) = split_in_face_legs_torso(input_features_copy)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = find_transformation(model_face, input_face)
    (input_transformed_torso, transformation_matrix_torso) =find_transformation(model_torso, input_torso)
    (input_transformed_legs, transformation_matrix_legs) = find_transformation(model_legs, input_legs)

    # Wrapped the transformed input in one whole pose
    input_transformation = unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)

    # In case of no normalisation, return here (ex; plotting)
    # Without normalisation the thresholds don't say anything
    #   -> so comparison is useless
    if(not normalise):
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)


    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.11 #0.065  of 0.11 ??
    rotation_tresh_torso = 40
    eucl_dis_tresh_legs = 0.055
    rotation_tresh_legs = 40

    eucld_dis_shoulders_tresh = 0.063
    ################################

    result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)

    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result


# TODO JOCHEN nieuwe versie single_person ??
def single_person_v2(model_features, input_features, normalise=True):
    # TODO: Make a local copy ??
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt naar callen van single_person()
    #model_features_copy = np.array(model_features)

    model_features_copy = copy.copy(model_features)

    # First, some safety checks ...
    # Each model must be a valid model, this means no Openpose errors (=a undetected body-part) are allowed
    #  -> models with undetected bodyparts are unvalid
    #  -> undetected body-parts are labeled by (0,0)
    '''
    if np.any(model_features_copy[:] == [0, 0]):
        for i in range(0,17):
            if model_features_copy[i][0] == 0 and model_features_copy[i][1] == 0:
                logger.warning(" Unvalid model pose, undetected body-parts")
                #result = MatchResult(False, error_score=0, input_transformation=None)
                #return result
    '''
    # Input is allowed to have a certain amount of undetected body parts
    # In that case, the corresponding point from the model is also changed to (0,0)
    #   -> afterwards matching can still proceed
    # The (0,0) points can't just be deleted because
    # because without them the featurearrays would become ambigu. (the correspondence between model and input)
    #
    # !! NOTE !! : the acceptation and introduction of (0,0) points
    # is a danger for our current normalisation
    # These particular origin points should not influence the normalisation
    # (which they do if we neglect them, xmin and ymin you know...)
    counter_not_found_points = 0
    if np.any(input_features[:] == [0,0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logger.warning(" Undetected body part in input: index(%d) %s", counter, prepocessing.get_bodypart(counter))
                if not (model_features_copy[counter][0] == 0 and model_features_copy[counter][1] == 0):
                    counter_not_found_points = counter_not_found_points+1
                model_features_copy[counter][0] = 0#np.nan
                model_features_copy[counter][1] = 0#np.nan
                #input_features[counter][0] = 0#np.nan
                #input_features[counter][1] = 0#np.nan

            counter = counter+1

    # if the input has more then 4 points not recognised then the model, then return false

    if counter_not_found_points > 4:
        logger.debug("Model has more feature then input therefore not matched")
        result = MatchResult(False,
                             error_score=0,
                             input_transformation=None)
        return result

    assert len(model_features_copy) == len(input_features)

    # Normalise features: crop => delen door Xmax & Ymax (NIEUWE MANIER!!)
    # !Note!: as state above, care should be taken when dealing
    #   with (0,0) points during normalisation
    #
    # TODO:
    # !Note2!: The exclusion of a feature in the torso-regio doesn't effect
    #   the affine transformation in the legs- and face-regio in general.
    #   BUT in some case it CAN influence the (max-)euclidean distance.
    #     -> (so could resolve in different MATCH result)
    #   This is the case when the undetected bodypart [=(0,0)] would be the
    #   minX or minY in the detected case.
    #   Now, in the absence of this minX or minY, another feature will deliver
    #   this value.
    #   -> The normalisation region is smaller and gives different values after normalisation.
    #
    #   (BV: als iemand met handen in zij staat maar de rechter ellenboog niet gedetect wordt
    #       => minX is nu van het rechthand dat in de zij staat.

    # TODO
    # It seems like the number of excluded features is proportional with the rotation angle
    # -> That is, the more features are missing, the higher the rotation angle becomes, this is weird

    if (normalise):
        model_features_copy = feature_scaling(model_features_copy)
        input_features = feature_scaling(input_features)
            # In case of no normalisation, return here (ex; plotting)
            # Without normalisation the thresholds don't say anything
            #   -> so comparison is useless
    else:
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    #Split features in three parts
    (model_face, model_torso, model_legs) = split_in_face_legs_torso_v2(model_features_copy)
    (input_face, input_torso, input_legs) = split_in_face_legs_torso_v2(input_features)

    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.098
    rotation_tresh_torso = 10.847
    eucl_dis_tresh_legs = 0.05
    rotation_tresh_legs = 14.527
    eucld_dis_shoulders_tresh = 0.085
    ################################
    #handle face
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face, input_face)
    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    if  np.count_nonzero(model_face)>5:
        if (np.count_nonzero(model_face) - np.count_nonzero(input_face)) < 2:
            result_face = True
        else:
            logger.debug("Model has more face feature then input therefore not matched")
            result_face = False
    else:
        logger.debug("too less points for face in model so face match")
        result_face = True

    #handle Torso
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso, input_torso)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)
    if  np.count_nonzero(model_torso)>5:
        if (np.count_nonzero(model_torso)-np.count_nonzero(input_torso)) < 2:
            result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                                       eucl_dis_tresh_torso, rotation_tresh_torso,
                                                                       max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)
        else:
            logger.debug("Model has more Torso feature then input therefore not matched")
            result_torso = False
    else:
        logger.debug("too less points for Torso in model so Torso match")
        result_torso = True

    #handle legs
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs, input_legs)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)
    if  np.count_nonzero(model_legs)>5:
        if (np.count_nonzero(model_legs) - np.count_nonzero(input_legs)) < 2:
            result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                                      eucl_dis_tresh_legs, rotation_tresh_legs)
        else:
            logger.debug("Model has more legs feature then input therefore not matched")
            result_legs = False
    else:
        logger.debug("too less points for legs in model so legs match")
        result_legs = True

    # Wrapped the transformed input in one whole pose
    input_transformation = unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)


    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs and result_face),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result

#Plot the calculated transformation on the model image
#And some other usefull plots for debugging
#NO NORMALIZING IS DONE HERE BECAUSE POINTS ARE PLOTTED ON THE ORIGINAL PICTURES!
def plot_single_person(model_features, input_features, model_image_name, input_image_name, input_title = "input",  model_title="model",
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


'''-----------------------------------------------------------'''
'''---------------------- MULTI PERSOOOOOOOON ---------------------'''
'''-----------------------------------------------------------'''

# -------------- GIlll ----------------------------
'''
Description multi_person():
This function is used in the first (simple) case: MODELS SEPARATELY (mutual orientation of different poses is not checked in this simple case)
    The human poses in the image have no relation with each other and they are considered separately 
    Foreach modelpose (in models_poses) a matching inputpose (in input_poses) is searched
    Only if for each modelpose matches with on of the inputposes in input_poses, a global match is achieved. 

Parameters:
@:param models_poses: Takes an array of models as input because every pose that needs to be mimic has it's own model
@:param input_poses: The input is one json file. This represents an image of multiple persons and they each try to mimic one of the poses in model

Returns:
@:returns False : in case GLOBAL MATCH FAILED
@:returns list_of_all_matches : (List of MatchCombo objects) each model 1 match with a input (this match is wrapped in a MatchCombo object)
'''
# THE NEW one: for every modelpose , a matching input is seeked
# Enkel zo kan je een GLOBAL MATCH FAILED besluiten na dat er geen matching inputpose is gevonden voor een modelpose
def multi_person(models_poses, input_poses, model_image_name, input_image_name):
    logger.info(" Multi-person matching...")
    logger.info(" amount of models: %d", len(models_poses))
    logger.info(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.error(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.error(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.warning(" !! WARNING !! Amount of input poses > model poses")
        #Continiue
        #return False

    # List of MatchCombo objects: each model has a 0 or 1 or more matches with a input (this match is wrapped in a MatchCombo object)
    #   -> we only allow the case with 1 match!
    #   1. cases with more than 1 matches are reduces with only 1 match (the best-match)

    #   2. case with 0 match result in a GLOBAL MATCH FAILED; because the modelpose is not found in the input
    #      If at the end of the whole matching iteration,
    #      there is a MatchCombo == None => a model is failed to match whith the possible inputposes SO global match failed
    #
    list_of_all_matches = []

    # The MatchCombo which links a modelpose with a matching inputpose
    # This is what we want to maximise ie: the best match of all possible matches found
    best_match_combo = None

    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = single_person(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)
                logger.info(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_input_pose = counter_input_pose + 1

        # If still no match is found (after looping over all the inputs); this model is not found in proposed inputposes
        # This can mean only one thing:
        #   1. The user(s) failed to mimic one of the proposed model poses

        if(best_match_combo  is None):
            logger.info(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1

    logger.info("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)

    return list_of_all_matches

'''
Description multi_person2()
This function is used in the second (complex) case: The models are dependent of each other in space
Their relation in space is checked in the same way as in case of single_pose(), 
    but now a affine transformation of the total of all poses is calculated

First a multi_pose() is executed and a list of best_matches is achieved
Then all separate input poses are combined into one input_pose_transformed
    This is the homography of all model poses displayed onto their best match inputpose.
    -> The modelpose is superimposed onto his matching inputpose
    This homography is calculated using only a translation and rotation, NO SCALING
Note that the input_transformed resulting from single_pose() is not used in this algorithm.

Final plots are only plotted if normalised is False
# DISCLAIMER on no-normalisation: 
# It's normal that the plot is fucked up in case of undetected body parts in the input
#  -> this is because no normalisation is done here (because of the plots)
#     and thus these sneaky (0,0) points are not handled.
# TODO: maybe not include the (0,0) handler only the normalising part??

A word on input poses with undetected body parts [ (0,0) points ]:
    Input poses with a certain amount of undetected body parts are allowed. 
    It is even so that,  if a best match is found for a model, 
    in the second step (procrustes) the undetected body parts 
    are overwritten with those of the model. 

Parameters:
@:param model_poses: Model containing multiple modelposes (one json file = one image because poses are seen as one whole) 
@:param input_poses: The input is one json file. This represents an image of multiple persons and 
                        together they try to mimic the whole model. 
@:param normalise: Default is True. In case of False; the max euclidean distance is calculated and reported
                    In case of True; the result in plotted on the images! 


Returns:
@:returns False : in case GLOBAL MATCH FAILED
@:returns True : Match! 
'''
#TODO fine-tune returns
#TODO optimaliseren voor geval van normalise! nu ist 2 in 1, ma voor productie is enkel normalise nodig in feite (ook ni helemaal waar -> feedback mss)
def multi_person2(model_poses, input_poses, model_image_name, input_image_name, normalise=True):
    # Find for each model_pose the best match input_pose
    # returns a list of best matches !! WITH normalisation !!
    # TODO fine-tune return tuple
    result = multi_person(model_poses, input_poses, model_image_name, input_image_name)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.error("Multi-person step1 match failed!")
        return False

    aantal_models = len(result)
    input_transformed_combined = np.zeros((18*aantal_models, 2))

    # The new input_transformed; contains all poses and wrapped in one total pose.
    # This input_transformed_combined is achieved by superimposing all the model poses on their corresponding inputpose
    input_transformed_combined = []

    updated_models_combined = []


    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    logger.info("-- multi_pose2(): looping over best-matches for procrustes:")
    for best_match in result:
        # First check for undetected body parts. If present=> make corresponding point in model also (0,0)
        # We can know strip them from our poses because we don't use split() for affine trans
        # TODO: deze clean updated_model_pose wordt eigenlijk al eens berekent in single_pose()
        #   -> loopke hier opnieuw is stevig redundant

        # make a array with the indecex of undetected points
        indexes_undetected_points = []  # todo stuk gil
        if np.any(best_match.input_features[:] == [0, 0]):
            assert True
            counter = 0
            for feature in best_match.input_features:
                if feature[0] == 0 and feature[1] == 0:  # (0,0)
                    indexes_undetected_points.append(counter)
                    #logger.warning(" Undetected body part in input: index(%d) %s", counter,prepocessing.get_bodypart(counter))
                    best_match.model_features[counter][0] = 0
                    best_match.model_features[counter][1] = 0
                counter = counter + 1


        best_match.input_features = best_match.input_features[( best_match.input_features[:, 0] != 0) & (best_match.input_features[:, 1] != 0)]
        best_match.model_features = best_match.model_features[( best_match.model_features[:, 0] != 0) & (best_match.model_features[:, 1] != 0)]
        # Note1: the input_transformed from single_pose() is not used!!!
        input_transformed = proc_do_it.superimpose(best_match.input_features, best_match.model_features, input_image_name, model_image_name)
        #input_transformed = proc_do_it.superimpose(best_match.input_features, best_match.model_features)



        input_transformed_combined.append(np.array(input_transformed))
        updated_models_combined.append(np.array(best_match.model_features))

        #logger.info("inputtt %s", str(input_transformed))
        #logger.info("modeelll %s ", str(best_match.model_features))

    assert len(input_transformed_combined) == len(model_poses)

    # TODO: harded code indexen weg doen
    # TODO: transpose van ne lijst? Mss beter toch met np.array() werken..  maar hoe init'en?
    # TODO : hier is wa refactoring/optimalisatie nodig ...

    #Lijst vervormen naar matrix

    input_transformed_combined = np.vstack([input_transformed_combined[0], input_transformed_combined[1]])
    #model_poses = np.vstack([model_poses[0], model_poses[1]])
    model_poses = np.vstack([updated_models_combined[0], updated_models_combined[1]])


    # Redundant, wordt enkel gebruikt voor plotten
    input_poses = np.vstack([input_poses[0], input_poses[1]])
    print("------trans: " , input_transformed_combined.shape)
    if(normalise):
        input_transformed_combined = feature_scaling(input_transformed_combined)
        model_poses = feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = find_transformation(model_poses, input_transformed_combined)

    # TODO return True in case of match
    if(normalise):
        max_eucl_distance = pose_comparison.max_euclidean_distance(model_poses, input_transformed_combined)
        logger.info("--->Max eucl distance: %s  (thresh ca. 0.13)", str(max_eucl_distance)) # torso thresh is 0.11

        markersize = 2

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        ax1.set_title('(input transformed (model superimposed on input )')
        ax1.plot(*zip(*input_transformed_combined), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax2.set_title('(model)')
        ax2.plot(*zip(*model_poses), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax3.set_title('(affine trans and model (red))')
        ax3.plot(*zip(*full_transformation), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
        ax3.plot(*zip(*model_poses), marker='o', color='b', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax = plt.gca()
        ax.invert_yaxis()
        #plt.show()
        plt.draw()


    else:
        logger.info("-- multi_pose2(): procrustes plotjes incoming ")
        plot_multi_pose(model_poses, input_poses, full_transformation,
                        model_image_name, input_image_name, "input poses", "full procrustes")
        plot_multi_pose(model_poses, input_transformed_combined, full_transformation,
                        model_image_name, input_image_name, "superimposed model on input", "full procrustes")

    #Block plots
    plt.show()
    return True




# ----------------- JOCHEN ---------------------------


def find_best_match1(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False


    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False


    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")

    list_of_all_matches = []
    model_poses = order_poses(model_poses)
    input_poses = order_poses(input_poses)
    best_match_combo = None
    used_poses = []

    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        if np.count_nonzero(model_pose)<9 or model_pose.size <9:
            counter_model_pose = counter_model_pose + 1
            logger.debug(" @@@@ bad model(%d) @@@@", counter_model_pose)
            continue

        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            # check if input pose has at least 8/2=4 points for transformation and that input pose isn't used twice.
            if np.count_nonzero(input_pose)<9 or input_pose.size <9 or used_poses.count(counter_input_pose) >0:
                counter_input_pose = counter_input_pose + 1
                logger.debug(" @@@@ bad input(%d) @@@@", counter_input_pose)
                continue

            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)

                logger.debug(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_input_pose = counter_input_pose + 1

        if(best_match_combo  is None):
            logger.debug(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        used_poses.append(best_match_combo.input_id)
        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1
    #end for loop
    logger.debug("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    '''
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)
    '''

    return list_of_all_matches

def multi_person3(model_poses, input_poses, normalise=True):

    result = find_best_match(model_poses, input_poses)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.debug("Multi-person step1 match failed!")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result


    aantal_models = len(result)
    input_transformed_combined = np.zeros((18*aantal_models, 2))

    input_transformed_combined = []

    updated_models_combined = []


    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    logger.debug("-- multi_pose(): looping over best-matches for procrustes:")

    for best_match in result:

        indexes_undetected_points = []
        if np.any(best_match.input_features[:] == [0, 0]):
            counter = 0
            for feature in best_match.input_features:
                if feature[0] == 0 and feature[1] == 0:  # (0,0)
                    indexes_undetected_points.append(counter)
                    #logger.warning(" Undetected body part in input: index(%d) %s", counter,prepocessing.get_bodypart(counter))
                    best_match.model_features[counter][0] = 0
                    best_match.model_features[counter][1] = 0
                counter = counter + 1

        (input_transformed,model) = proc_do_it.superimpose(best_match.input_features, best_match.model_features)

        input_transformed_combined.append(np.array(input_transformed))
        updated_models_combined.append(np.array(model))


    assert len(input_transformed_combined) == len(updated_models_combined)
    #not enough corresponding points
    if not (len(input_transformed_combined) >0 ):
        logger.debug("not enough corresponding points between model and input")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

    input_transformed_combined = np.vstack(input_transformed_combined)
    model_poses =np.vstack(updated_models_combined)


    if(normalise):
        input_transformed_combined = normalising.feature_scaling(input_transformed_combined)
        model_poses = normalising.feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = affine_transformation.find_transformation(model_poses, input_transformed_combined)

    return (True,0,0)

def order_poses(poses):
    ordered = []
    for i in range(0,len(poses)):
        if(np.nonzero(poses[i]) > 8):
            pose= poses[i][:,0]
            placed = False
            place_counter = 1
            while not placed:
                place = i - place_counter
                if (place > -1):
                    prev_pose = poses[place][:,0]
                    try:
                        if np.min(pose[np.nonzero(pose)]) < np.min(prev_pose[np.nonzero(prev_pose)]):
                            place_counter = place_counter+1
                        else:
                            ordered.insert(1,poses[i])
                            placed = True
                    except ValueError:
                        placed =True
                else:
                    ordered.insert(0,poses[i])
                    placed = True
    return ordered

def find_ordered_matches(model_poses,input_poses):
    if(len(input_poses)== 0 or len(model_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    if(len(input_poses) < len(model_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    if (len(input_poses) > len(model_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")

    model_poses = order_poses(model_poses)
    input_poses = order_poses(input_poses)
    model_pose = model_poses[0]
    matches = []

    #find first match of poses
    for model_counter in range(0,len(model_poses)):
        model_pose = model_poses[model_counter]
        matches.append([])
        match_found = False
        start_input =0
        if model_counter >0:
            start_input =  matches[model_counter-1][0]
        for input_counter in range(0,len(input_poses)):
            input_pose = input_poses[input_counter]
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)
            if result_match:
                match_found = True
                matches[model_counter].append(input_counter)
        if match_found == False:
            logger.debug("no match found for model %d", model_counter)
            return False

    logger.debug("matches found %s"," ".join(str(e) for e in matches))
    return matches

def multi_person_ordered(model_poses, input_poses, normalise=True):

    matches = find_ordered_matches(model_poses,input_poses)
    if matches == False:
        return MatchResult(False, error_score=0, input_transformation=None)
    #np = np.array(matches)
    possiblities = cartesian(matches)

    return MatchResult(True, error_score=0, input_transformation=None)

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


# TODO van jochen   MAG WEG??  -> duplicate van in z_multuperson_match ( is oude multi_person() van gil aangepast)
def find_best_match2(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")
        #Continiue
        #return False

    # List of MatchCombo objects: each model has a 0 or 1 or more matches with a input (this match is wrapped in a MatchCombo object)
    #   -> we only allow the case with 1 match!
    #   1. cases with more than 1 matches are reduces with only 1 match (the best-match)

    #   2. case with 0 match result in a GLOBAL MATCH FAILED; because the modelpose is not found in the input
    #      If at the end of the whole matching iteration,
    #      there is a MatchCombo == None => a model is failed to match whith the possible inputposes SO global match failed
    #
    list_of_all_matches = []

    # The MatchCombo which links a modelpose with a matching inputpose
    # This is what we want to maximise ie: the best match of all possible matches found
    best_match_combo = None
    used_poses = []
    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        if np.count_nonzero(model_pose)<9 or model_pose.size <9:
            counter_model_pose = counter_model_pose + 1
            logger.debug(" @@@@ bad model(%d) @@@@", counter_model_pose)
            continue

        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            # check if input pose has at least 8/2=4 points for transformation and that input pose isn't used twice.
            if np.count_nonzero(input_pose)<9 or input_pose.size <9 or used_poses.count(counter_input_pose) >0:
                counter_input_pose = counter_input_pose + 1
                logger.debug(" @@@@ bad input(%d) @@@@", counter_input_pose)
                continue

            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)

                logger.debug(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_input_pose = counter_input_pose + 1

        # If still no match is found (after looping over all the inputs); this model is not found in proposed inputposes
        # This can mean only one thing:
        #   1. The user(s) failed to mimic one of the proposed model poses

        if(best_match_combo  is None):
            logger.debug(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        used_poses.append(best_match_combo.input_id)
        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1
    #end for loop
    logger.debug("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    '''
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)
    '''

    return list_of_all_matches

'''
Description order_matches()
try to find all model poses from left to right in input poses.
'''
# TODO ook van Jochen: MAG WEG??  -> duplicate van in z_multuperson_match  ranschikken van poses van links naar rechts ??
def order_matches(models_poses, input_poses):
    logger.debug(" Multi-person matching...")
    logger.debug(" amount of models: %d", len(models_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    # Some safety checks first ..
    if(len(input_poses)== 0 or len(models_poses) == 0):
        logger.debug(" Multi person match failed. Inputposes or modelposes are empty")
        return False

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if(len(input_poses) < len(models_poses)):
        logger.debug(" Multi person match failed. Amount of input poses < model poses")
        return False

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(models_poses)):
        logger.debug(" !! WARNING !! Amount of input poses > model poses")
        #Continiue
        #return False

    # List of MatchCombo objects: each model has a 0 or 1 or more matches with a input (this match is wrapped in a MatchCombo object)
    #   -> we only allow the case with 1 match!
    #   1. cases with more than 1 matches are reduces with only 1 match (the best-match)

    #   2. case with 0 match result in a GLOBAL MATCH FAILED; because the modelpose is not found in the input
    #      If at the end of the whole matching iteration,
    #      there is a MatchCombo == None => a model is failed to match whith the possible inputposes SO global match failed
    #
    list_of_all_matches = []

    # The MatchCombo which links a modelpose with a matching inputpose
    # This is what we want to maximise ie: the best match of all possible matches found
    best_match_combo = None
    used_poses = []
    # Iterate over the model poses
    # TODO: improve search algorithm (not necessary i guess, as it is only illustrative)
    counter_model_pose = 1
    logger.debug(" ->Searching a best-match for each model in the modelposes ...")
    for model_pose in models_poses:
        if np.count_nonzero(model_pose)<9 or model_pose.size <9:
            counter_model_pose = counter_model_pose + 1
            logger.debug(" @@@@ bad model(%d) @@@@", counter_model_pose)
            continue

        logger.debug(" Iterate for modelpose(%d)", counter_model_pose)
        counter_input_pose = 1
        for input_pose in input_poses:
            # check if input pose has at least 8/2=4 points for transformation and that input pose isn't used twice.
            if np.count_nonzero(input_pose)<9 or input_pose.size <9 or used_poses.count(counter_input_pose) >0:
                counter_input_pose = counter_input_pose + 1
                logger.debug(" @@@@ bad input(%d) @@@@", counter_input_pose)
                continue

            logger.debug(" @@@@ Matching model(%d) with input(%d) @@@@", counter_model_pose, counter_input_pose)
            # Do single pose matching
            (result_match, error_score, input_transformation) = singleperson_match.single_person_v2(model_pose, input_pose, True)

            if (result_match):
                match_combo = MatchCombo(error_score, counter_input_pose, counter_model_pose,model_pose, input_pose, input_transformation)

                logger.debug(" Match: %s ModelPose(%d)->InputPose(%d)", result_match, counter_model_pose, counter_input_pose)

                # If current MatchCombo object is empty, init it
                if best_match_combo is None:
                    best_match_combo = match_combo
                # If new match is better (=has a lower error_score) than current best_match, overwrite
                elif best_match_combo.error_score > match_combo.error_score:
                    best_match_combo = match_combo

            counter_input_pose = counter_input_pose + 1

        # If still no match is found (after looping over all the inputs); this model is not found in proposed inputposes
        # This can mean only one thing:
        #   1. The user(s) failed to mimic one of the proposed model poses

        if(best_match_combo  is None):
            logger.debug(" MATCH FAILED. No match found for modelpose(%d). User failed to match a modelpose ", counter_model_pose)
            return False

        used_poses.append(best_match_combo.input_id)
        # After comparing every possible models with a inputpose, append to match_list
        list_of_all_matches.append(best_match_combo)
        # And reset best_match field for next iteration
        best_match_combo = None

        counter_model_pose = counter_model_pose + 1
    #end for loop
    logger.debug("-- multi_pose1(): looping over best-matches for producing plotjes:")
    # Plotjes: affine transformation is calculated again but now without normalisation
    '''
    for i in list_of_all_matches:
        if i is not None:
            (result, error_score, input_transformation) = single_person(i.model_features, i.input_features, False)
            plot_match(i.model_features, i.input_features, input_transformation, model_image_name, input_image_name)
    '''

    return list_of_all_matches

# TODO nog nen andere mutli_person van jochen MAG WEG??  -> duplicate van in z_multuperson_match
def multi_person33(model_poses, input_poses, normalise=True):
    # Find for each model_pose the best match input_pose
    # returns a list of best matches !! WITH normalisation !!
    # TODO fine-tune return tuple
    result = find_best_match(model_poses, input_poses)

    if(result is False):
        # Minimum one model pose is not matched with a input pose
        logger.debug("Multi-person step1 match failed!")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result


    aantal_models = len(result)
    input_transformed_combined = np.zeros((18*aantal_models, 2))

    # The new input_transformed; contains all poses and wrapped in one total pose.
    # This input_transformed_combined is achieved by superimposing all the model poses on their corresponding inputpose
    input_transformed_combined = []

    updated_models_combined = []


    # Loop over the best-matches
    #       [modelpose 1 -> inputpose x ; modelpose2 -> inputpose y; ...]
    logger.debug("-- multi_pose(): looping over best-matches for procrustes:")

    for best_match in result:
        # First check for undetected body parts. If present=> make corresponding point in model also (0,0)
        # We can know strip them from our poses because we don't use split() for affine trans
        # TODO: deze clean updated_model_pose wordt eigenlijk al eens berekent in single_pose()
        #   -> loopke hier opnieuw is stevig redundant

        # make a array with the indecex of undetected points
        indexes_undetected_points = []
        if np.any(best_match.input_features[:] == [0, 0]):
            assert True
            counter = 0
            for feature in best_match.input_features:
                if feature[0] == 0 and feature[1] == 0:  # (0,0)
                    indexes_undetected_points.append(counter)
                    #logger.warning(" Undetected body part in input: index(%d) %s", counter,prepocessing.get_bodypart(counter))
                    best_match.model_features[counter][0] = 0
                    best_match.model_features[counter][1] = 0
                counter = counter + 1




        (input_transformed,model) = proc_do_it.superimpose(best_match.input_features, best_match.model_features)

        input_transformed_combined.append(np.array(input_transformed))
        updated_models_combined.append(np.array(model))

        #logger.info("inputtt %s", str(input_transformed))
        #logger.info("modeelll %s ", str(best_match.model_features))

    assert len(input_transformed_combined) == len(updated_models_combined)
    #not enough corresponding points
    if not (len(input_transformed_combined) >0 ):
        logger.debug("not enough corresponding points between model and input")
        result = MatchResult(False, error_score=0, input_transformation=None)
        return result

    # TODO: harded code indexen weg doen
    # TODO: transpose van ne lijst? Mss beter toch met np.array() werken..  maar hoe init'en?
    # TODO : hier is wa refactoring/optimalisatie nodig ...

    #Lijst vervormen naar matrix

    input_transformed_combined = np.vstack(input_transformed_combined)
    #model_poses = np.vstack([model_poses[0], model_poses[1]])
    model_poses =np.vstack(updated_models_combined)


    if(normalise):
        input_transformed_combined = feature_scaling(input_transformed_combined)
        model_poses = feature_scaling(model_poses)

    # Calc the affine trans of the whole
    (full_transformation, A_matrix) = affine_transformation.find_transformation(model_poses, input_transformed_combined)
    '''
    # TODO return True in case of match
    if(normalise):
        max_eucl_distance = pose_comparison.max_euclidean_distance(model_poses, input_transformed_combined)
        logger.info("--->Max eucl distance: %s  (thresh ca. 0.13)", str(max_eucl_distance)) # torso thresh is 0.11

        markersize = 2

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        ax1.set_title('(input transformed (model superimposed on input )')
        ax1.plot(*zip(*input_transformed_combined), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax2.set_title('(model)')
        ax2.plot(*zip(*model_poses), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

        ax3.set_title('(affine trans and model (red))')
        ax3.plot(*zip(*full_transformation), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
        ax3.plot(*zip(*model_poses), marker='o', color='b', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax = plt.gca()
        ax.invert_yaxis()
        #plt.show()
        plt.draw()


    else:
        logger.info("-- multi_pose2(): procrustes plotjes incoming ")
        plot_multi_pose(model_poses, input_poses, full_transformation,
                        model_image_name, input_image_name, "input poses", "full procrustes")
        plot_multi_pose(model_poses, input_transformed_combined, full_transformation,
                        model_image_name, input_image_name, "superimposed model on input", "full procrustes")

    #Block plots
    plt.show()
'''
    return (True,0,0)

#Plots all Three: model, input and transformation
def plot_multi_pose(model_features, input_features, full_transform, model_image_name, input_image_name, text_input, text_transform):
    # plot vars
    markersize = 2

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    #ax1.legend(handles=[red_patch])

    ax2.set_title('('+text_input+')')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)
    #ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('('+text_transform+')')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*full_transform), marker='o', color='blue', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])
    plt.draw()
    #plt.show()

    return

#Plots all Three: model, input and transformation
def plot_match(model_features, input_features, input_transform_features, model_image_name, input_image_name):
    # plot vars
    markersize = 2

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    #ax1.legend(handles=[red_patch])


    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)
    #ax2.legend(handles=[mpatches.Patch(color='blue', label='input')])

    ax3.set_title('Transformed input on model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transform_features), marker='o', color='blue', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])
    plt.draw()
    #plt.show()

    return