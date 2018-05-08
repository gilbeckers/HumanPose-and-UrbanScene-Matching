import heapq
import logging
import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from urbanscene.features import  max_euclidean_distance, euclidean_distance
import common
import thresholds
import random
import plot_vars

def perspective_correction(H2, p_model, p_input, model_pose_features, input_pose_features, model_img, input_img, plot=False):
    # we assume input_pose and model_pose contain same amount of features, as we would also expected in this stage of pipeline

    h, w = input_img.shape
    h = round(h * 6 / 5)
    w = round(w * 6 / 5)

    perspective_transform_input = cv2.warpPerspective(input_img, H2, (w, h)) # persp_matrix2 transforms input onto model_plane
    if plot:
        plt.figure()
        plt.subplot(221), plt.imshow(model_img), plt.title('Model')
        plt.subplot(222), plt.imshow(perspective_transform_input), plt.title('Perspective transformed Input')
        plt.subplot(223), plt.imshow(input_img), plt.title('Input')
        plt.show(block=False)

    my_input_pts2 = np.float32(p_input).reshape(-1, 1, 2)  # bit reshapeing so the cv2.perspectiveTransform() works
    p_input_persp_trans = cv2.perspectiveTransform(my_input_pts2, H2)  # transform(input_pts_2D)
    p_input_persp_trans = np.squeeze(p_input_persp_trans[:])  # strip 1 dimension

    max_euclidean_error = max_euclidean_distance(p_model, p_input_persp_trans)
    logging.debug('PERSSPECTIVE 1: max error: %d', max_euclidean_error)

    #TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(p_model)
    input_features_trans_norm = common.feature_scaling(p_input_persp_trans)

    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug('PERSSPECTIVE NORM 1: max error: %f', max_euclidean_error)

    # -- 2E MANIERRR: normaliseren enkel de pose
    input_pose_trans = p_input_persp_trans[len(p_input_persp_trans) - len(input_pose_features): len(
        p_input_persp_trans)]  # niet perse perspective corrected, hangt af van input
    model_pose_norm = common.feature_scaling(model_pose_features)
    input_pose_trans_norm = common.feature_scaling(input_pose_trans)

    max_euclidean_error = max_euclidean_distance(model_pose_norm, input_pose_trans_norm)

    logging.debug('PERSSPECTIVE NORM 2: max error: %f', max_euclidean_error)

    markersize = 3
    # model_img_arr = np.asarray(model_img)
    # input_img_arr = np.asarray(input_img)
    # input_persp_img_arr = np.asarray()

    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        #ax1.imshow(model_img)
        ax1.imshow(np.asarray(model_img), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("Model ") #kever17.jpg
        #ax1.plot(*zip(*p_model), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose_features), marker='o', color='red', ls='', label='pose', ms=markersize )  # ms = markersize
        #red_patch = mpatches.Patch(color='magenta', label='model')
        #ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("Input ") #kever10.jpg
        ax2.imshow(np.asarray(input_img), cmap='gray')
        #ax2.plot(*zip(*p_input), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose_features), marker='*', color='r', ls='', ms=markersize)
        #ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("Perspectief herstelde input")
        ax3.imshow(np.asarray(perspective_transform_input), cmap='gray')
        ax3.plot(*zip(*input_pose_trans), marker='o', color='b', ls='', ms=markersize)
        #ax3.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])

        # ax4.set_title("trans-input onto model")
        # ax4.imshow(np.asarray(model_img), cmap='gray')
        # ax4.plot(*zip(*p_input_persp_trans), marker='o', color='b', ls='', ms=markersize)
        # ax4.plot(*zip(*p_model), marker='o', color='magenta', ls='', ms=markersize)
        # ax4.plot(*zip(*model_pose_features), marker='o', color='green', ls='', ms=markersize)
        # ax4.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])
        # plt.tight_layout()
        plt.show(block=False)

    return (p_input_persp_trans, input_pose_trans,  perspective_transform_input)

def affine_multi(p_model_good, p_input_good, model_pose, input_pose, model_image_height, model_image_width,input_image_h, input_image_w, label, model_img, persp_input_img, input_pose_org, plot=False):

    # include some random features of background:
    #model_pose = np.vstack((model_pose, p_model_good[0], p_model_good[1],p_model_good[2], p_model_good[3],p_model_good[4], p_model_good[5]))
    #input_pose = np.vstack((input_pose, p_input_good[0], p_input_good[1],p_input_good[2], p_input_good[3],p_input_good[4], p_input_good[5]))
    model_pose_org = np.copy(model_pose)
    random_features = random.sample(range(0, len(p_model_good)), thresholds.AMOUNT_BACKGROUND_FEATURES)
    model_pose = [np.array(model_pose)]
    input_pose = [np.array(input_pose)]
    logging.debug("THE RANDOM FEATURES: %s", str(random_features))

    # include some random features of background:
    for i in random_features:
        model_pose.append(np.array(p_model_good[i]))
        input_pose.append(np.array(p_input_good[i]))

    model_pose = np.vstack(model_pose)
    input_pose = np.vstack(input_pose)


    (input_transformed, M) = common.find_transformation(model_pose, input_pose)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed = unpad(np.dot(pad(np.vstack((p_input_good, input_pose))), M))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?

    # Norm manier 1: pose normaliseren
    #model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_pose)))
    #input_features_trans_norm = common.feature_scaling(input_transformed)

    # Norm manier 2: image_resolution normaliseren
    model_features_norm = common.scene_feature_scaling(np.vstack((p_model_good, model_pose)), model_image_width, model_image_height)
    #input_features_trans_norm = common.scene_feature_scaling(input_transformed, input_image_w, input_image_h)
    input_features_trans_norm = common.scene_feature_scaling(input_transformed, model_image_width, model_image_height)

    # f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(14, 6))
    # # ax1.set_title(model_image_name + ' (model)')
    # ax1.set_title("model norm")
    # ax1.plot(*zip(*model_features_norm), marker='o', color='magenta', ls='', label='model',
    #          ms=3)
    # ax1.plot(*zip(*input_features_trans_norm), marker='o', color='r', ls='', ms=3)

    euclidean_error_norm = euclidean_distance(model_features_norm, input_features_trans_norm)
    #  index 2(rechts) en 5(links) zijn de polsen
    #logging.warning("Distance polsen  links: %f   rechts: %f", round(euclidean_error_torso_norm[2], 3), round(euclidean_error_torso_norm[5], 3) )
    logging.debug("#### AFFINE RAND NORM Sum TOTAL: %f" , sum(euclidean_error_norm))
    logging.debug("#### AFFINE RAND NORM Sum TOTAL/#matches: %f", sum(euclidean_error_norm)/len(p_model_good))
    max_euclidean_error_norm = max(euclidean_error_norm)#max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_total: %f", max_euclidean_error_norm)

    max_euclidean_error = max_euclidean_distance(np.vstack((p_model_good, model_pose)), input_transformed)

    logging.debug("#### AFFINE RAND " + label + "  error_torso: %f", max_euclidean_error)

    markersize = 3
    fs= 8  #fontsize
    if plot:
        # --- First row plot ---
        plain_input_img = cv2.imread(plot_vars.input_path, cv2.IMREAD_GRAYSCALE)
        f = plt.figure(figsize=(10, 8))


        f.suptitle("US matching | score="+ str(round(max_euclidean_error_norm,4)) + " (thresh=ca " + str(thresholds.AFFINE_TRANS_WHOLE_DISTANCE) +" )", fontsize=10)
        plt.subplot(2, 2, 1)
        plt.imshow(np.asarray(plain_input_img), cmap='gray')
        plt.title("input: " + plot_vars.input_name, fontsize=fs)
        plt.plot(*zip(*input_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        plain_model_img = cv2.imread(plot_vars.model_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 2, 2)
        plt.imshow(np.asarray(plain_model_img), cmap='gray')
        plt.title("model: " + plot_vars.model_name, fontsize=fs)
        plt.plot(*zip(*model_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        # --- Second row plot ---
        #f.set_figheight(20)
        plt.subplot(2, 3, 4)
        plt.imshow(np.asarray(persp_input_img), cmap='gray')
        plt.axis('off')
        plt.title("corrected input", fontsize=fs)
        plt.plot(*zip(*p_input_good), marker='o', color='r', label='features', ls='', ms=markersize)
        plt.plot(*zip(*input_pose), marker='o', color='blue', label='pose+randfeat', ls='', ms=markersize)
        plt.legend(fontsize=fs - 1)
        #plt.legend(handles=[mpatches.Patch(color='red', label='features'),mpatches.Patch(color='blue', label='pose')])


        plt.subplot(2, 3, 5)
        plt.imshow(np.asarray(plain_model_img), cmap='gray')
        plt.title("model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='features',ms=markersize)  # ms = markersize
        plt.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='pose+randfeat',ms=markersize)  # ms = markersize
        #red_patch = mpatches.Patch(color='magenta', label='model')
        #plt.legend(handles=[red_patch])
        plt.legend(fontsize=fs - 1)

        plt.subplot(2, 3, 6)
        plt.imshow(np.asarray(model_img), cmap='gray')
        plt.title("transform on model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*np.vstack((p_model_good, model_pose))), marker='o', color='magenta', ls='',
                 label='model',
                 ms=markersize-1)  # ms = markersize
        plt.plot(*zip(*input_transformed), marker='o', color='aqua', ls='', label='input',
                 ms=markersize-1)  # ms = markersize
        #plt.legend(handles=[mpatches.Patch(color='green', label='trans-input'), mpatches.Patch(color='magenta', label='model')])
        plt.legend(fontsize=fs-1)

        if plot_vars.write_img:
            plot_name= plot_vars.model_name.split(".")[0] + "_" + plot_vars.input_name.split(".")[0]
            plt.savefig('./plots/'+plot_name+'.png')

        #f, axes = plt.subplots(2, )
        #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
        # implot = ax1.imshow(np.asarray(model_img), cmap='gray')
        # # ax1.set_title(model_image_name + ' (model)')
        # ax1.set_title("model")
        # ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # red_patch = mpatches.Patch(color='magenta', label='model')
        # ax1.legend(handles=[red_patch])
        #
        # # ax2.set_title(input_image_name + ' (input)')
        # ax2.set_title("input")
        # ax2.imshow(np.asarray(input_img), cmap='gray')
        # ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
        # ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        # ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
        #
        # ax3.set_title("aff split() " + label)
        # ax3.imshow(np.asarray(model_img), cmap='gray')
        # ax3.plot(*zip(*np.vstack((p_model_good, model_pose))), marker='o', color='magenta', ls='',
        #          label='model',
        #          ms=markersize)  # ms = markersize
        # ax3.plot(*zip(*input_transformed), marker='o', color='green', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # ax3.legend(handles=[mpatches.Patch(color='green', label='trans-input'),
        #                     mpatches.Patch(color='magenta', label='model')])
        #
        # # plt.tight_layout()
        # #plt.show(block=False)

    return max_euclidean_error_norm






def affine_trans_interaction_both(p_model_good, p_input_good, model_pose, input_pose,  model_img, input_img, label):
    #input_pose = p_input_good[len(p_input_good) - size_pose: len(p_input_good)]  # niet perse perspective corrected, hangt af van input
    #model_pose = p_model_good[len(p_model_good) - size_pose: len(p_input_good)]

    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    (input_transformed_torso, M_tor) = common.find_transformation(np.vstack((p_model_good, model_torso)),np.vstack((p_input_good, input_torso)))
    (input_transformed_legs, M_legs) = common.find_transformation(np.vstack((p_model_good, model_legs)),np.vstack((p_input_good, input_legs)))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error)
    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = common.feature_scaling(input_transformed_legs)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM" + label + "  error_legs: ", max_euclidean_error)


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_img), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_img), cmap='gray')
    ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split" + label)
    ax3.imshow(np.asarray(model_img), cmap='gray')
    ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*np.vstack((input_transformed_torso, input_transformed_legs))), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input torso'),
                        mpatches.Patch(color='magenta', label='model')])

    # plt.tight_layout()
    plt.show(block=False)
    return None

# enkel A berekenen uit pose features lijkt mij het logischte want enkel de pose kan varieeren in ratio
# de scene niet aangezien die ratio's normaal vast zijn!!
def affine_trans_interaction_only_pose(p_model_good, p_input_good, model_pose, input_pose, model_img, input_img, label):
    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    #model_torso = np.vstack((model_torso, p_model_good[0], p_model_good[1], p_model_good[10] ))
    #input_torso = np.vstack((input_torso, p_input_good[0], p_input_good[1], p_input_good[10]))

    #model_legs = np.vstack((model_legs, p_model_good[0], p_model_good[1], p_model_good[10] ))
    #input_legs = np.vstack((input_legs, p_input_good[0], p_input_good[1], p_input_good[10]))


    (input_transformed_torso, M_tor) = common.find_transformation(model_torso,input_torso)
    (input_transformed_legs, M_legs) = common.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((p_input_good, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((p_input_good, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)

    max_euclidean_error_torso = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error_torso)


    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = common.feature_scaling(input_transformed_legs)

    max_euclidean_error_legs = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_legs: ", max_euclidean_error_legs)

    if max_euclidean_error_torso < 0.15 and max_euclidean_error_legs < 0.15:
        print("#### MATCH!!!  ###")
    else:
        print("#### NO MATCH!! ###")


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 2
    ms_pose = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_img), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=ms_pose)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_img), cmap='gray')
    ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=ms_pose)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split " + label)
    ax3.imshow(np.asarray(model_img), cmap='gray')
    ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed_legs), marker='o', color='green', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed_torso), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input torso'),
                        mpatches.Patch(color='magenta', label='model')])

    # plt.tight_layout()
    plt.show(block=False)
    return None

def affine_trans_interaction_pose_rand_scene(p_model_good, p_input_good, model_pose, input_pose,  model_img, input_img, label, plot=False):
    # TODO: Deze geeft (momenteel betere resultaten dan den _normalise => verschillen tussen matches en niet-matches is pak groter
    #TODO: normalising van whole !! en niet normaliseren van legs en torso appart
    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    model_torso = np.vstack((model_torso, p_model_good[0], p_model_good[1], p_model_good[10] ))
    input_torso = np.vstack((input_torso, p_input_good[0], p_input_good[1], p_input_good[10]))

    model_legs = np.vstack((model_legs, p_model_good[0], p_model_good[1], p_model_good[10] ))
    input_legs = np.vstack((input_legs, p_input_good[0], p_input_good[1], p_input_good[10]))


    (input_transformed_torso, M_tor) = common.find_transformation(model_torso, input_torso)
    (input_transformed_legs, M_legs) = common.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((p_input_good, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((p_input_good, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)

    euclidean_error_torso_norm = euclidean_distance(model_features_norm, input_features_trans_norm)
    #  index 2(rechts) en 5(links) zijn de polsen
    #logging.warning("Distance polsen  links: %f   rechts: %f", round(euclidean_error_torso_norm[2], 3), round(euclidean_error_torso_norm[5], 3) )
    logging.debug("#### AFFINE RAND NORM Sum torso: %f" , sum(euclidean_error_torso_norm))
    max_euclidean_error_torso_norm = max(euclidean_error_torso_norm)#max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_torso: %f", max_euclidean_error_torso_norm)

    second_max = heapq.nlargest(2, euclidean_error_torso_norm)
    logging.debug("#### AFFINE RAND NORM 2e MAX torso: %f", second_max[1])




    model_features_norm = common.feature_scaling(np.vstack((p_model_good, model_legs)))
    input_features_trans_norm = common.feature_scaling(input_transformed_legs)

    euclidean_error_legs_norm = euclidean_distance(model_features_norm, input_features_trans_norm)
    max_euclidean_error_legs_norm = max(euclidean_error_legs_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_legs: %f", max_euclidean_error_legs_norm)

    # if max_euclidean_error_torso_norm < thresh and max_euclidean_error_legs_norm < thresh:
    #     logging.debug("#### MATCH!!!  ###")
    #     match = True
    # else:
    #     logging.debug("#### NO MATCH!! ###")
    #     match = False

    max_euclidean_error_torso = max_euclidean_distance(np.vstack((p_model_good, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((p_model_good, model_legs)), input_transformed_legs)

    logging.debug("#### AFFINE RAND " + label + "  error_torso: %f", max_euclidean_error_torso)
    logging.debug("#### AFFINE RAND " + label + "  error_legs: %f", max_euclidean_error_legs)


    markersize = 3
    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        implot = ax1.imshow(np.asarray(model_img), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("model")
        ax1.plot(*zip(*p_model_good), marker='o', color='magenta', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
                 ms=markersize)  # ms = markersize
        red_patch = mpatches.Patch(color='magenta', label='model')
        ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("input")
        ax2.imshow(np.asarray(input_img), cmap='gray')
        ax2.plot(*zip(*p_input_good), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("aff split() " + label)
        ax3.imshow(np.asarray(model_img), cmap='gray')
        ax3.plot(*zip(*np.vstack((p_model_good, model_torso, model_legs))), marker='o', color='magenta', ls='',
                 label='model',
                 ms=markersize)  # ms = markersize
        ax3.plot(*zip(*input_transformed_legs), marker='o', color='green', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax3.plot(*zip(*input_transformed_torso), marker='o', color='blue', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax3.legend(handles=[mpatches.Patch(color='blue', label='trans-input torso'),
                            mpatches.Patch(color='green', label='trans-input legs'),
                            mpatches.Patch(color='magenta', label='model')])

        # plt.tight_layout()
        plt.show(block=False)
    return (max_euclidean_error_torso_norm, max_euclidean_error_legs_norm, sum(euclidean_error_torso_norm), sum(euclidean_error_legs_norm))



