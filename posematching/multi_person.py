from posematching.single_person import MatchResult, MatchCombo, match_single
from common import handle_undetected_points, feature_scaling, find_transformation
from posematching.procrustes import superimpose
from posematching.pose_comparison import max_euclidean_distance
import numpy as np
import logging
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logger = logging.getLogger("multi_person")


def match(model_poses, input_poses, normalise=True):
    logger.debug(" amount of models: %d", len(model_poses))
    logger.debug(" amount of inputs: %d", len(input_poses))

    # Some safety checks first ..
    if (len(input_poses) == 0 or len(model_poses) == 0):
        logger.debug("FAIL: Multi person match failed. Inputposes or modelposes are empty")
        return MatchResult(False, error_score=0, input_transformation=None)

    # Then check if there are equal or more input poses than model poses
    # When the amount of input poses is smaller than model poses a global match FAILED is decided
    # Note: amount of models may be smaller than amount of input models:
    #       -> ex in case of people on the background which are also detected by openpose => background noise
    #       -> TODO: ? foreground detection necessary (in case less models than there are inputs)  NOT SURE ...
    if (len(input_poses) < len(model_poses)):
        logger.debug("FAIL: Multi person match failed. Amount of input poses < model poses")
        return MatchResult(False, error_score=0, input_transformation=None)

    # When there are more inputposes than modelsposes, print warning
    # I this case pose match still needs to proceed,
    #  BUT it's possible that a background pose is matched with one of the proposed modelposes!! (something we don't want)
    if (len(input_poses) > len(model_poses)):
        logger.warning(" !! WARNING !! Amount of input poses > model poses")

    matches_dict = find_ordered_matches(model_poses,input_poses)

    logger.debug("matches found %s", str(matches_dict))
    if matches_dict == False:
        logger.debug("FAIL: No ordered matches found")
        return MatchResult(False, error_score=0, input_transformation=None)

    #if not contains_every_input_once(matches_dict):
    if input_pose_solo_multiple_times(matches_dict, len(input_poses)):
        logger.debug("FAIL: An inputpose is linked to multiple modelpose as only possible match => global fail (injection remember)")
        return MatchResult(False, error_score=0, input_transformation=None)



    # source : https://stackoverflow.com/questions/38721847/python-generate-all-combination-from-values-in-dict-of-lists
    allInputs = sorted(matches_dict)
    combinations = list(itertools.product(*(matches_dict[Name] for Name in allInputs)))
    result_permuations = {}
    for permutation in combinations:
        if len(permutation) != len(set(permutation)):  # check for duplicate inputs bv (1,1,0) => inputpose 1 wordt gelinkt aan modelpose0 en modelpose1
            logger.warning("LOCAL FAIL: SAME INPUTPOSES MAPPED ON MULTIPLE MODELPOSES")
            continue

        input_transformed_combined = []
        updated_models_combined = []
        for model_index, input_index_val in enumerate(permutation):
            (input_pose, model_pose) = handle_undetected_points(input_poses[input_index_val], model_poses[model_index])
            (input_transformed, model) = superimpose(input_pose, model_pose)

            input_transformed_combined.append(np.array(input_transformed))
            updated_models_combined.append(np.array(model))

        assert len(input_transformed_combined) == len(updated_models_combined)

        # not enough corresponding points  #TODO voor wa is dit?
        if not (len(input_transformed_combined) > 0):
            logger.debug("FAIL: not enough corresponding points between model and input")
            result = MatchResult(False, error_score=0, input_transformation=None)
            return result

        # Calc the affine trans of the whole
        input_transformed_combined = np.vstack(input_transformed_combined)
        updated_models_combined = np.vstack(updated_models_combined)

        if (normalise):
            input_transformed_combined = feature_scaling(input_transformed_combined)
            updated_models_combined = feature_scaling(updated_models_combined)
        logger.debug("shape model_pose %s", str(updated_models_combined.shape))
        (full_transformation, A_matrix) = find_transformation(updated_models_combined, input_transformed_combined)
        max_eucl_distance = max_euclidean_distance(updated_models_combined, full_transformation)
        result_permuations[permutation] = max_eucl_distance
        logger.info("----> Matching for permutation %s  | Max eucl distance: %s  (thresh ca. 0.13)",permutation, str(max_eucl_distance))  # torso thresh is 0.11


    # TODO: nog max nemen van resultaat.
    print("result scores: " , result_permuations)
    return MatchResult(True, error_score=0, input_transformation=None)




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
            return False
        matches_dict[model_counter] = model_matches


    #logger.debug("matches found %s"," ".join(str(e) for e in matches))
    #return list(itertools.product(*matches))
    #return matches
    return matches_dict


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

    # Load images
    model_image = plt.imread(model_image_name)
    input_image = plt.imread(input_image_name)

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
    ax3.legend(handles=[mpatches.Patch(color='magenta', label='Model'), mpatches.Patch(color='blue', label='Input transformed')])
    plt.draw()
    #plt.show()

    return