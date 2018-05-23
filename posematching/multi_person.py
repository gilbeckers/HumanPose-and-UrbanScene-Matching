from posematching.single_person import MatchResult, MatchCombo, match_single, MatchResultMulti
from common import handle_undetected_points, feature_scaling, find_transformation
from posematching.procrustes import superimpose, superimpose_old
from posematching.pose_comparison import max_euclidean_distance
import numpy as np
import logging
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import thresholds
import plot_vars
logger = logging.getLogger("multi_person")


def match(model_poses, input_poses, plot=False, input_image = None, model_image=None, normalise=True):
    logger.debug(" amount of models: %d", len(model_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))
    plot_vars.amount_input_persons = len(input_poses)
    plot_vars.amount_model_persons = len(model_poses)

    model_poses = np.copy(model_poses)
    input_poses = np.copy(input_poses)

    # Some safety checks first ..
    if (len(input_poses) == 0 or len(model_poses) == 0):
        logger.debug("FAIL: Multi person match failed. Inputposes or modelposes are empty")
        return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None)

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if (len(input_poses) < len(model_poses)):
        logger.debug("FAIL: Multi person match failed. Amount of input poses < model poses")
        return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None)

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(model_poses)):
        logger.warning(" !! WARNING !! Amount of input poses > model poses")

    #
    if len(model_poses)==1 and len(input_poses)==1:
        logger.debug("Modelpose and inputpose both contain only one pose, so performing simple single_pose matching")
        match_result_single = match_single(model_poses[0], input_poses[0], True)

        if plot:
            plot_match(model_poses[0], input_poses[0], match_result_single.input_transformation, model_image, input_image)
        # !!! BELANGRIJK!!  wordt ook gedaan in match_single(), maar maakt copy, dus hier opnieuw doen!!
        (model_pose, input_pose) = handle_undetected_points(model_poses[0], input_poses[0])

        if match_result_single.match_bool:
            result_single = {}
            result_single[0] = {
                "score": match_result_single.error_score,
                "model": model_pose, #model_poses[0],
                # "model":updated_models_combined_nonorm,
                "input": input_pose, #input_poses[0]
                # "input": input_transformed_combined_nonorm
            }
            return MatchResultMulti(match_result_single.match_bool,
                                    error_score=match_result_single.error_score,
                                    input_transformation=match_result_single.input_transformation, #TODO niet meer gebruikt??
                                    matching_permutations=result_single)
        else:
            return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None )

    matches_dict, ordered_model_poses, ordered_input_poses = find_ordered_matches(model_poses,input_poses)

    logger.debug("matches found %s", str(matches_dict))
    if matches_dict == None:
        logger.debug("FAIL: No ordered matches found")
        return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None)
    else: # override poses with the ordered ones
        model_poses = ordered_model_poses
        input_poses = ordered_input_poses

    if input_pose_solo_multiple_times(matches_dict, len(input_poses)):
        logger.debug("FAIL: An inputpose is linked to multiple modelpose as only possible match => global fail (injection remember)")
        #return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None)



    # source : https://stackoverflow.com/questions/38721847/python-generate-all-combination-from-values-in-dict-of-lists
    allInputs = sorted(matches_dict)
    combinations = list(itertools.product(*(matches_dict[Name] for Name in allInputs)))
    result_permuations = {}
    for permutation in combinations:
        if len(permutation) != len(set(permutation)):  # check for duplicate inputs bv (1,1,0) => inputpose 1 wordt gelinkt aan modelpose0 en modelpose1
            logger.warning("--> Matching permutation %s : FAIL: ONE INPUTPOSE MAPPED ON MULTIPLE MODELPOSES (no injection)", permutation)
            continue

        input_transformed_combined = []
        updated_models_combined = []

        unchanged_input = []
        unchanged_model = []
        for model_index, input_index_val in enumerate(permutation):
            logger.debug("Superimposing for model %d  and input %d", model_index, input_index_val)
            (input_pose, model_pose) = handle_undetected_points(input_poses[input_index_val], model_poses[model_index])
            (input_transformed, model) = superimpose_old(input_pose, model_pose, plot=False, input_image= None, model_image=None)

            input_transformed_combined.append(np.array(input_transformed))
            updated_models_combined.append(np.array(model))

            unchanged_input.append(np.array(input_pose))
            unchanged_model.append(np.array(model_pose))

        # logger.debug("model: %s" , str(updated_models_combined))
        # logger.debug("input: %s" , str(input_transformed_combined))

        assert len(input_transformed_combined) == len(updated_models_combined)

        # not enough corresponding points  #TODO voor wa is dit?
        if not (len(input_transformed_combined) > 0):
            logger.debug("--> Matching permutation %s : FAIL: not enough corresponding points between model and input", permutation)
            result = MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=None)
            return result

        # Calc the affine trans of the whole
        input_transformed_combined = np.vstack(input_transformed_combined)
        updated_models_combined = np.vstack(updated_models_combined)

        unchanged_input = np.vstack(unchanged_input)
        unchanged_model = np.vstack(unchanged_model)

        input_transformed_combined_nonorm = input_transformed_combined
        updated_models_combined_nonorm =updated_models_combined

        if (normalise):
            input_transformed_combined = feature_scaling(input_transformed_combined)
            updated_models_combined = feature_scaling(updated_models_combined)

        logger.debug("shape model_pose %s", str(updated_models_combined.shape))
        (full_transformation, A_matrix) = find_transformation(updated_models_combined, input_transformed_combined)
        max_eucl_distance = max_euclidean_distance(updated_models_combined, full_transformation)

        if plot:
            (full_transformation_nonorm, A_matrix) = find_transformation(updated_models_combined_nonorm, input_transformed_combined_nonorm)
            plot_match(updated_models_combined_nonorm, input_transformed_combined_nonorm, full_transformation_nonorm, model_image, input_image)
            plot_match(unchanged_model,unchanged_input, unchanged_model)


        if max_eucl_distance<=thresholds.MP_DISCTANCE:
            result_permuations[permutation] = {
                "score" : max_eucl_distance ,
                "model" : unchanged_model,
                #"model":updated_models_combined_nonorm,
                "input" : unchanged_input
                #"input": input_transformed_combined_nonorm
            }
            logger.info("--> MATCH! permutation %s  | Max distance: %0.4f  (thresh %0.4f)", permutation,
                        max_eucl_distance,thresholds.MP_DISCTANCE )
        else:
            logger.info("--> NO-MATCH! permutation %s  | Max distance: %0.4f  (thresh %0.4f)", permutation,
                        max_eucl_distance, thresholds.MP_DISCTANCE)
    # TODO: nog max nemen van resultaat.
    #logger.debug("result scores: %s" , result_permuations)

    # If result_permutations is still empty, no match was found so return false
    if result_permuations:
        return MatchResultMulti(True, error_score=0, input_transformation=None, matching_permutations=result_permuations)

    return MatchResultMulti(False, error_score=0, input_transformation=None, matching_permutations=result_permuations)


def find_ordered_matches(model_poses,input_poses):

    model_poses = order_poses(model_poses)
    input_poses = order_poses(input_poses)

    matches = []
    matches_dict = {}

    #find first match of poses
    for model_counter in range(0,len(model_poses)):
        model_pose = model_poses[model_counter]

        matches.append([])
        model_matches = []

        match_found = False
        start_input =0
        if model_counter >0:
            start_input =  matches[model_counter-1][0]
        for input_counter in range(start_input,len(input_poses)):
            input_pose = input_poses[input_counter]
            # Do single pose matching
            (result_match, error_score, input_transformation) = match_single(model_pose, input_pose, True)
            logging.debug("model%d & input%d | result: %s  score %f ", model_counter, input_counter, str(result_match), round(error_score, 4))
            if result_match:
                match_found = True
                matches[model_counter].append(input_counter)

                model_matches.append(input_counter)
        if match_found == False:
            logger.debug("no match found for model %d", model_counter)
            return (None, None, None)
        matches_dict[model_counter] = model_matches


    #logger.debug("matches found %s"," ".join(str(e) for e in matches))
    #return list(itertools.product(*matches))
    #return matches
    return matches_dict, model_poses, input_poses


# Check if an inputpose is linked to multiple modelpose as only possible match.
# When this occurs, in order to have a global match, this would mean one inputpose would have to match with 2 different modelposes
# This is ofcours not possible because the input-model relations is a injection(one-to-one)
def input_pose_solo_multiple_times(matches_dict, input_length):
    solo_inputpose = np.zeros(input_length)
    for model_index, matching_inputs in matches_dict.items():
        if len(matching_inputs) == 1:
            solo_inputpose[matching_inputs[0]] = solo_inputpose[matching_inputs[0]]+1

    if np.argmax(solo_inputpose>1): # contains a element greater than 1 => a single input is linked to multiple models as only possible match

        return True

    return False

def order_poses(poses):
    ordered = []
    for i in range(0,len(poses)):
        if np.count_nonzero(poses[i]) > 8:
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







def plot_match(model_features, input_features, input_transform_features, model_image_name, input_image_name):
    # plot vars

    markersize = 2

    # Load images , if model_image_name is of type string => it's a path, so read the image in mem
    if isinstance(model_image_name, str):
        model_image = plt.imread(model_image_name)
        input_image = plt.imread(input_image_name)
    else:
        model_image = model_image_name  # plt.imread(model_image_name)
        input_image = input_image_name  # plt.imread(input_image_name)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')

    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)

    ax3.set_title('Transformed input on model')
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transform_features), marker='o', color='blue', ls='', ms=markersize)
    ax3.plot(*input_transform_features[17], marker='x', color='red', ls='', ms=4)
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])
    #plt.draw()
    plt.show()

    return

def plot_poses(model_features, input_features, model_image_name, input_image_name):
    # plot vars
    markersize = 2

    # Load images , if model_image_name is of type string => it's a path, so read the image in mem
    if isinstance(model_image_name, str):
        model_image = plt.imread(model_image_name)
        input_image = plt.imread(input_image_name)
    else:
        model_image = model_image_name  # plt.imread(model_image_name)
        input_image = input_image_name  # plt.imread(input_image_name)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    ax1.set_title('(model)')
    ax1.plot(*zip(*model_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')

    ax2.set_title('(input)')
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_features), marker='o', color='red', ls='', ms=markersize)



    return

