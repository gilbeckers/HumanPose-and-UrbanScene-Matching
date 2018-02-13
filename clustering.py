import cv2
import numpy as np


# TODO: kan momenteel maar max 2 gebouwen in afbeelding handelenen!!

# TODO: wat als 2 gebouwen inderdaad heel dicht bij elkaar staan? en dus centers inderdaad dicht bij elkaar liggen -- onder theshold dus--
# Returns an array. It dependents on the centers of the clusters in how many groups the features are split
# Returns a boolean which is true when image contains only one building described by features
def kmean(model_pts_2D, input_pts_2D):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    model_ret, model_label, model_center = cv2.kmeans(model_pts_2D, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    input_ret, input_label, input_center = cv2.kmeans(input_pts_2D, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    print("center : ", model_center)
    distance_centers_model = np.sqrt(
        (model_center[0][0] - model_center[1][0]) ** 2 + (model_center[0][1] - model_center[1][1]) ** 2)
    print("distance ceners: ", distance_centers_model)
    distance_centers_input = np.sqrt(
        (input_center[0][0] - input_center[1][0]) ** 2 + (input_center[0][1] - input_center[1][1]) ** 2)

    # TODO eerst normaliseren ofcourzz
    if (distance_centers_model <= 170):  # als centers te dicht bij elkaar liggen => opniew knn met k=1
        print("-$$$$$$$$$$$$-Centers te dicht bij elkaar!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return (np.array([[model_pts_2D], [input_pts_2D]]), True)  # return an list where input and model are each 1 group of features

    #Else, apply mask; separate the data and form 2 feature groups
    #
    model_A = model_pts_2D[model_label.ravel() == 0]
    model_B = model_pts_2D[model_label.ravel() == 1]

    input_A = input_pts_2D[model_label.ravel() == 0]
    input_B = input_pts_2D[model_label.ravel() == 1]

    return (np.array([[model_A, model_B], [input_A, input_B]]), False) # return an list where input and model are each 2 groups of features
    #return (model_ret, model_label, model_center, input_ret, input_label, input_center)

