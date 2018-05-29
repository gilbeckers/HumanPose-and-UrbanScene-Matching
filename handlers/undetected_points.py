import numpy as np

def handle_model_only_policy(input_features, model_features):
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt na callen van single_person()
    # model_features_copy = np.array(model_features)
    model_features_copy = model_features.copy()
    input_features_copy = input_features.copy()


    # In this second version, the model is allowed to have undetected features
    if np.any(model_features[:] == [0, 0]):
        counter = 0
        for feature in model_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logging.debug(" Undetected body part in MODEL: index(%d) %s", counter,get_bodypart(counter))
                input_features_copy[counter][0] = 0
                input_features_copy[counter][1] = 0
            counter = counter + 1

    assert len(model_features_copy) == len(input_features_copy)
    return (input_features_copy, model_features_copy,counter)

def handle_all_policy(input_features, model_features):

    model_features_copy = model_features.copy()
    input_features_copy = input_features.copy()

    if np.any(input_features[:] == [0, 0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logger.debug(" Undetected body part in input: index(%d) %s", counter,get_bodypart(counter))
                model_features_copy[counter][0] = 0
                model_features_copy[counter][1] = 0
                # input_features[counter][0] = 0#np.nan
                # input_features[counter][1] = 0#np.nan
            counter = counter + 1

    # In this second version, the model is allowed to have undetected features
    if np.any(model_features[:] == [0, 0]):
        counter = 0
        for feature in model_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                #logging.debug(" Undetected body part in MODEL: index(%d) %s", counter,get_bodypart(counter))
                input_features_copy[counter][0] = 0
                input_features_copy[counter][1] = 0
            counter = counter + 1

    assert len(model_features_copy) == len(input_features_copy)

    return (input_features_copy, model_features_copy)
