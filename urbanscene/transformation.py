import heapq
import logging
import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
#import common
from handlers import undetected_points
from handlers import transformation
from handlers import scaling
from handlers import compare
import thresholds
import random
import plot_vars

def perspective_correction(H2, good_model_scene_features, good_input_scene_features, model_pose_features, input_pose_features, model_image, input_image, plot=False):
    # we assume input_pose and model_pose contain same amount of features, as we would also expected in this stage of pipeline

    input_image_height, input_image_width = input_image.shape
    input_image_height = round(input_image_height * 6 / 5)
    input_image_width = round(input_image_width * 6 / 5)

    perspective_transform_input = cv2.warpPerspective(input_image, H2, (input_image_width, input_image_height)) # persp_matrix2 transforms input onto model_plane
    if plot:
        plt.figure()
        plt.subplot(221), plt.imshow(model_image), plt.title('Model')
        plt.subplot(222), plt.imshow(perspective_transform_input), plt.title('Perspective transformed Input')
        plt.subplot(223), plt.imshow(input_image), plt.title('Input')
        plt.show(block=False)

    my_input_pts2 = np.float32(good_input_scene_features).reshape(-1, 1, 2)  # bit reshapeing so the cv2.perspectiveTransform() works
    good_input_scene_features_persp_trans = cv2.perspectiveTransform(my_input_pts2, H2)  # transform(input_pts_2D)
    good_input_scene_features_persp_trans = np.squeeze(good_input_scene_features_persp_trans[:])  # strip 1 dimension

    max_euclidean_error = compare.max_euclidean_distance(good_model_scene_features, good_input_scene_features_persp_trans)
    logging.debug('PERSSPECTIVE 1: max error: %d', max_euclidean_error)

    #TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = scaling.feature_scaling(good_model_scene_features)
    input_features_trans_norm = scaling.feature_scaling(good_input_scene_features_persp_trans)

    max_euclidean_error = compare.max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug('PERSSPECTIVE NORM 1: max error: %f', max_euclidean_error)

    # -- 2E MANIERRR: normaliseren enkel de pose
    input_pose_trans = good_input_scene_features_persp_trans[len(good_input_scene_features_persp_trans) - len(input_pose_features): len(
        good_input_scene_features_persp_trans)]  # niet perse perspective corrected, hangt af van input
    model_pose_norm = scaling.feature_scaling(model_pose_features)
    input_pose_trans_norm = scaling.feature_scaling(input_pose_trans)

    max_euclidean_error = compare.max_euclidean_distance(model_pose_norm, input_pose_trans_norm)

    logging.debug('PERSSPECTIVE NORM 2: max error: %f', max_euclidean_error)

    markersize = 3
    # model_image_arr = np.asarray(model_image)
    # input_image_arr = np.asarray(input_image)
    # input_persp_img_arr = np.asarray()

    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        #ax1.imshow(model_image)
        ax1.imshow(np.asarray(model_image), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("Model ") #kever17.jpg
        #ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose_features), marker='o', color='red', ls='', label='pose', ms=markersize )  # ms = markersize
        #red_patch = mpatches.Patch(color='magenta', label='model')
        #ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("Input ") #kever10.jpg
        ax2.imshow(np.asarray(input_image), cmap='gray')
        #ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose_features), marker='*', color='r', ls='', ms=markersize)
        #ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("Perspectief herstelde input")
        ax3.imshow(np.asarray(perspective_transform_input), cmap='gray')
        ax3.plot(*zip(*input_pose_trans), marker='o', color='b', ls='', ms=markersize)
        #ax3.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])

        # ax4.set_title("trans-input onto model")
        # ax4.imshow(np.asarray(model_image), cmap='gray')
        # ax4.plot(*zip(*good_input_scene_features_persp_trans), marker='o', color='b', ls='', ms=markersize)
        # ax4.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', ms=markersize)
        # ax4.plot(*zip(*model_pose_features), marker='o', color='green', ls='', ms=markersize)
        # ax4.legend(handles=[mpatches.Patch(color='blue', label='corrected input')])
        # plt.tight_layout()
        plt.show()


    return (good_input_scene_features_persp_trans, input_pose_trans,  perspective_transform_input)

def affine_multi(good_model_scene_features, good_input_scene_features, model_pose, input_pose, label, model_image, input_image, input_pose_org, plot=False):
    if len(good_model_scene_features) < thresholds.AMOUNT_BACKGROUND_FEATURES:
        return np.inf  # hoge euclidische score


    model_image_height =  model_image.shape[0]
    model_image_width =  model_image.shape[1]
    input_image_height =  input_image.shape[0]
    input_image_width =  input_image.shape[1]

    # include some random features of background:
    model_pose_org = np.copy(model_pose)
    random_features = random.sample(range(0, len(good_model_scene_features)), thresholds.AMOUNT_BACKGROUND_FEATURES)
    model_pose = [np.array(model_pose)]
    input_pose = [np.array(input_pose)]
    logging.debug("THE RANDOM FEATURES: %s", str(random_features))

    # include some random features of background:
    for i in random_features:
        model_pose.append(np.array(good_model_scene_features[i]))
        input_pose.append(np.array(good_input_scene_features[i]))

    #voeg min en max toe
    # model_pose.append(good_model_scene_features[np.argmax(good_model_scene_features[:, 0])])
    # model_pose.append(good_model_scene_features[np.argmin(good_model_scene_features[:, 0])])
    # model_pose.append(good_model_scene_features[np.argmax(good_model_scene_features[:, 1])])
    # model_pose.append(good_model_scene_features[np.argmin(good_model_scene_features[:, 1])])
    # #random de som van beide
    # model_pose.append(good_model_scene_features[np.argmax(sum(good_model_scene_features))])
    # model_pose.append(good_model_scene_features[np.argmin(sum(good_model_scene_features))])
    # #voeg mediaan toe van x en y
    # good_model_scene_features = good_model_scene_features[good_model_scene_features[:,0].argsort()]
    # model_pose.append(good_model_scene_features[len(good_model_scene_features)//2])
    # model_pose.append(good_model_scene_features[len(good_model_scene_features)//4])
    # model_pose.append(good_model_scene_features[(len(good_model_scene_features)//4)*3])
    #
    # good_model_scene_features = good_model_scene_features[good_model_scene_features[:,1].argsort()]
    # model_pose.append(good_model_scene_features[len(good_model_scene_features)//2])
    # model_pose.append(good_model_scene_features[len(good_model_scene_features)//4])
    # model_pose.append(good_model_scene_features[(len(good_model_scene_features)//4)*3])
    #
    #
    #
    # input_pose.append(good_input_scene_features[np.argmax(good_input_scene_features[:, 0])])
    # input_pose.append(good_input_scene_features[np.argmin(good_input_scene_features[:, 0])])
    # input_pose.append(good_input_scene_features[np.argmax(good_input_scene_features[:, 1])])
    # input_pose.append(good_input_scene_features[np.argmin(good_input_scene_features[:, 1])])

    # input_pose.append(good_input_scene_features[np.argmax(sum(good_input_scene_features))])
    # input_pose.append(good_input_scene_features[np.argmin(sum(good_input_scene_features))])

    # good_input_scene_features = good_input_scene_features[good_input_scene_features[:,0].argsort()]
    # input_pose.append(good_input_scene_features[len(good_input_scene_features)//2])
    # input_pose.append(good_input_scene_features[len(good_input_scene_features)//4])
    # input_pose.append(good_input_scene_features[(len(good_input_scene_features)//4)*3])
    #
    # good_input_scene_features = good_input_scene_features[good_input_scene_features[:,1].argsort()]
    # input_pose.append(good_input_scene_features[len(good_input_scene_features)//2])
    # input_pose.append(good_input_scene_features[len(good_input_scene_features)//4])
    # input_pose.append(good_input_scene_features[(len(good_input_scene_features)//4)*3])
    model_pose = np.vstack(model_pose)
    input_pose = np.vstack(input_pose)


    (input_transformed, M) = transformation.find_transformation(model_pose, input_pose)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_pose))), M))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?

    # Norm manier 1: pose normaliseren
    #model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_pose)))
    #input_features_trans_norm = common.feature_scaling(input_transformed)

    # Norm manier 2: image_resolution normaliseren
    model_features_norm = scaling.scene_feature_scaling(np.vstack((good_model_scene_features, model_pose)), model_image_width, model_image_height)
    #input_features_trans_norm = scaling.scene_feature_scaling(input_transformed, input_image_width, input_image_height)
    input_features_trans_norm = scaling.scene_feature_scaling(input_transformed, model_image_width, model_image_height)

    # f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(14, 6))
    # # ax1.set_title(model_image_name + ' (model)')
    # ax1.set_title("model norm")
    # ax1.plot(*zip(*model_features_norm), marker='o', color='magenta', ls='', label='model',
    #          ms=3)
    # ax1.plot(*zip(*input_features_trans_norm), marker='o', color='r', ls='', ms=3)

    euclidean_error_norm = compare.euclidean_distance(model_features_norm, input_features_trans_norm)
    #  index 2(rechts) en 5(links) zijn de polsen
    #logging.warning("Distance polsen  links: %f   rechts: %f", round(euclidean_error_torso_norm[2], 3), round(euclidean_error_torso_norm[5], 3) )
    logging.debug("#### AFFINE RAND NORM Sum TOTAL: %f" , sum(euclidean_error_norm))
    logging.debug("#### AFFINE RAND NORM Sum TOTAL/#matches: %f", sum(euclidean_error_norm)/len(good_model_scene_features))
    max_euclidean_error_norm = max(euclidean_error_norm)#max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_total: %f", max_euclidean_error_norm)

    max_euclidean_error = compare.max_euclidean_distance(np.vstack((good_model_scene_features, model_pose)), input_transformed)

    logging.debug("#### AFFINE RAND " + label + "  error_torso: %f", max_euclidean_error)

    markersize = 3
    fs= 8  #fontsize
    if plot:
        # --- First row plot ---
        plain_input_image = cv2.imread(plot_vars.input_path, cv2.IMREAD_GRAYSCALE)
        f = plt.figure(figsize=(10, 8))


        f.suptitle("US matching | score="+ str(round(max_euclidean_error_norm,4)) + " (thresh=ca " + str(thresholds.AFFINE_TRANS_WHOLE_DISTANCE) +" )", fontsize=10)
        plt.subplot(2, 2, 1)
        plt.imshow(np.asarray(plain_input_image), cmap='gray')
        plt.title("input: " + plot_vars.input_name, fontsize=fs)
        plt.plot(*zip(*input_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        plain_model_image = cv2.imread(plot_vars.model_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 2, 2)
        plt.imshow(np.asarray(plain_model_image), cmap='gray')
        plt.title("model: " + plot_vars.model_name, fontsize=fs)
        plt.plot(*zip(*model_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        # --- Second row plot ---
        #f.set_figheight(20)
        plt.subplot(2, 3, 4)
        plt.imshow(np.asarray(input_image), cmap='gray')
        plt.axis('off')
        plt.title("corrected input", fontsize=fs)
        plt.plot(*zip(*good_input_scene_features), marker='o', color='r', label='features', ls='', ms=markersize)
        plt.plot(*zip(*input_pose), marker='o', color='blue', label='pose+randfeat', ls='', ms=markersize)
        plt.legend(fontsize=fs - 1)
        #plt.legend(handles=[mpatches.Patch(color='red', label='features'),mpatches.Patch(color='blue', label='pose')])


        plt.subplot(2, 3, 5)
        plt.imshow(np.asarray(plain_model_image), cmap='gray')
        plt.title("model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='features',ms=markersize)  # ms = markersize
        plt.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='pose+randfeat',ms=markersize)  # ms = markersize
        #red_patch = mpatches.Patch(color='magenta', label='model')
        #plt.legend(handles=[red_patch])
        plt.legend(fontsize=fs - 1)

        plt.subplot(2, 3, 6)
        plt.imshow(np.asarray(model_image), cmap='gray')
        plt.title("transform on model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*np.vstack((good_model_scene_features, model_pose))), marker='o', color='magenta', ls='',
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
        # implot = ax1.imshow(np.asarray(model_image), cmap='gray')
        # # ax1.set_title(model_image_name + ' (model)')
        # ax1.set_title("model")
        # ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # red_patch = mpatches.Patch(color='magenta', label='model')
        # ax1.legend(handles=[red_patch])
        #
        # # ax2.set_title(input_image_name + ' (input)')
        # ax2.set_title("input")
        # ax2.imshow(np.asarray(input_image), cmap='gray')
        # ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
        # ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        # ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
        #
        # ax3.set_title("aff split() " + label)
        # ax3.imshow(np.asarray(model_image), cmap='gray')
        # ax3.plot(*zip(*np.vstack((good_model_scene_features, model_pose))), marker='o', color='magenta', ls='',
        #          label='model',
        #          ms=markersize)  # ms = markersize
        # ax3.plot(*zip(*input_transformed), marker='o', color='green', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # ax3.legend(handles=[mpatches.Patch(color='green', label='trans-input'),
        #                     mpatches.Patch(color='magenta', label='model')])
        #
        # # plt.tight_layout()
        plt.show()

    return max_euclidean_error_norm



def affine_multi_important_posefeat(good_model_scene_features, good_input_scene_features, model_pose, input_pose, model_image_height, model_image_width,input_image_height, input_image_width, label, model_image, input_image, input_pose_org,pose_feat=4, plot=False):

    model_pose_org = np.copy(model_pose)
    random_features = random.sample(range(0, len(good_model_scene_features)), thresholds.AMOUNT_BACKGROUND_FEATURES)
    model_pose = [np.array(model_pose)]
    input_pose = [np.array(input_pose)]
    #logging.debug("THE RANDOM FEATURES: %s", str(random_features))

    # include some random features of background:
    for i in random_features:
        model_pose.append(np.array(good_model_scene_features[i]))
        input_pose.append(np.array(good_input_scene_features[i]))

    model_pose = np.vstack(model_pose)
    input_pose = np.vstack(input_pose)

    (input_transformed, M) = transformation.find_transformation(model_pose, input_pose)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_pose))), M))

    # Norm manier 2: image_resolution normaliseren
    model_features_norm = scaling.scene_feature_scaling(np.vstack((good_model_scene_features, model_pose)), model_image_width, model_image_height)
    #input_features_trans_norm = common.scene_feature_scaling(input_transformed, input_image_width, input_image_height)
    input_features_trans_norm = scaling.scene_feature_scaling(input_transformed, model_image_width, model_image_height)

    euclidean_error_norm = compare.euclidean_distance(model_features_norm, input_features_trans_norm)
    max_euclidean_error_norm = max(euclidean_error_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_total: %f", max_euclidean_error_norm)

    dis_model = distPoseAndBackgroundFeat(model_features_norm, pose_feat)
    dis_trans_input = distPoseAndBackgroundFeat(input_features_trans_norm, pose_feat)

    #logging.info("distance MODEL : %s"  , str(dis_model) )
    #logging.info("distance INPUT : %s", str(dis_trans_input))

    max_dis = np.max(np.abs(dis_model-dis_trans_input))
    max_index = np.argmax(np.abs(dis_model-dis_trans_input))
    logging.info("distance DIFF : %s  index: %s" ,str(max_dis), str(max_index))

    max_euclidean_error = max_euclidean_distance(np.vstack((good_model_scene_features, model_pose)), input_transformed)

    markersize = 3
    fs= 8  #fontsize
    if plot:
        # --- First row plot ---
        plain_input_image = cv2.imread(plot_vars.input_path, cv2.IMREAD_GRAYSCALE)
        f = plt.figure(figsize=(10, 8))


        f.suptitle("US matching | score="+ str(round(max_euclidean_error_norm,4)) + " (thresh=ca " + str(thresholds.AFFINE_TRANS_WHOLE_DISTANCE) +" )", fontsize=10)
        plt.subplot(2, 2, 1)
        plt.imshow(np.asarray(plain_input_image), cmap='gray')
        plt.title("input: " + plot_vars.input_name, fontsize=fs)
        plt.plot(*zip(*input_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        plain_model_image = cv2.imread(plot_vars.model_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 2, 2)
        plt.imshow(np.asarray(plain_model_image), cmap='gray')
        plt.title("model: " + plot_vars.model_name, fontsize=fs)
        plt.plot(*zip(*model_pose_org), marker='o', color='blue', label='pose', ls='', ms=markersize-1)

        # --- Second row plot ---
        #f.set_figheight(20)
        plt.subplot(2, 3, 4)
        plt.imshow(np.asarray(input_image), cmap='gray')
        plt.axis('off')
        plt.title("corrected input", fontsize=fs)
        plt.plot(*zip(*good_input_scene_features), marker='o', color='r', label='features', ls='', ms=markersize)
        plt.plot(*zip(*input_pose), marker='o', color='blue', label='pose+randfeat', ls='', ms=markersize)
        plt.legend(fontsize=fs - 1)
        #plt.legend(handles=[mpatches.Patch(color='red', label='features'),mpatches.Patch(color='blue', label='pose')])


        plt.subplot(2, 3, 5)
        plt.imshow(np.asarray(plain_model_image), cmap='gray')
        plt.title("model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='features',ms=markersize)  # ms = markersize
        plt.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='pose+randfeat',ms=markersize)  # ms = markersize
        #red_patch = mpatches.Patch(color='magenta', label='model')
        #plt.legend(handles=[red_patch])
        plt.legend(fontsize=fs - 1)

        plt.subplot(2, 3, 6)
        plt.imshow(np.asarray(model_image), cmap='gray')
        plt.title("transform on model", fontsize=fs)
        plt.axis('off')
        plt.plot(*zip(*np.vstack((good_model_scene_features, model_pose))), marker='o', color='magenta', ls='',
                 label='model',
                 ms=markersize-1)  # ms = markersize
        plt.plot(*zip(*input_transformed), marker='o', color='aqua', ls='', label='input',
                 ms=markersize-1)  # ms = markersize

        plt.plot(*model_pose[4,:], marker='x', color='gold', ls='', label='input',
                 ms=7, linewidth=4.0)  # ms = markersize
        plt.plot(*good_model_scene_features[max_index-18, :], marker='x', color='gold', ls='', label='input',
                 ms=7, linewidth=4.0)  # ms = markersize

        plt.plot(*input_transformed[4, :],'bs', marker='x', ls='', label='input',ms=7, linewidth=4.0)  # ms = markersize
        plt.plot(*input_transformed[max_index, :], marker='x', color='r', ls='', label='input',
                 ms=7, linewidth=4.0)  # ms = markersize

        #plt.plot(*input_transformed[max_index,:], marker='x', color='gold', ls='', label='input',ms=5)  # ms = markersize
        #plt.legend(handles=[mpatches.Patch(color='green', label='trans-input'), mpatches.Patch(color='magenta', label='model')])
        plt.legend(fontsize=fs-1)

        if plot_vars.write_img:
            plot_name= plot_vars.model_name.split(".")[0] + "_" + plot_vars.input_name.split(".")[0]
            plt.savefig('./plots/'+plot_name+'.png')

        #f, axes = plt.subplots(2, )
        #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
        # implot = ax1.imshow(np.asarray(model_image), cmap='gray')
        # # ax1.set_title(model_image_name + ' (model)')
        # ax1.set_title("model")
        # ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
        #          ms=markersize)  # ms = markersize
        # red_patch = mpatches.Patch(color='magenta', label='model')
        # ax1.legend(handles=[red_patch])
        #
        # # ax2.set_title(input_image_name + ' (input)')
        # ax2.set_title("input")
        # ax2.imshow(np.asarray(input_image), cmap='gray')
        # ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
        # ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        # ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
        #
        # ax3.set_title("aff split() " + label)
        # ax3.imshow(np.asarray(model_image), cmap='gray')
        # ax3.plot(*zip(*np.vstack((good_model_scene_features, model_pose))), marker='o', color='magenta', ls='',
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


def dist(x,y):
    return np.sqrt( np.sum((x-y)**2, axis=1))
# Calc distance between most important pose-feature and all background features
def distPoseAndBackgroundFeat(features, pose_feat):
    lengt = features.shape[0]
    # Make an array with as every row the pose_feature
    pose_feat_arr = np.ones((features.shape[0],2))
    pose_feat_arr[:,0] = pose_feat_arr[:,0]* features[pose_feat][0]
    pose_feat_arr[:, 1] = pose_feat_arr[:, 1] * features[pose_feat][1]
    dis = dist(pose_feat_arr, features)

    return  dis












def affine_trans_interaction_both(good_model_scene_features, good_input_scene_features, model_pose, input_pose,  model_image, input_image, label):
    #input_pose = good_input_scene_features[len(good_input_scene_features) - size_pose: len(good_input_scene_features)]  # niet perse perspective corrected, hangt af van input
    #model_pose = good_model_scene_features[len(good_model_scene_features) - size_pose: len(good_input_scene_features)]

    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    (input_transformed_torso, M_tor) = common.find_transformation(np.vstack((good_model_scene_features, model_torso)),np.vstack((good_input_scene_features, input_torso)))
    (input_transformed_legs, M_legs) = common.find_transformation(np.vstack((good_model_scene_features, model_legs)),np.vstack((good_input_scene_features, input_legs)))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error)
    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_legs)))
    input_features_trans_norm = common.feature_scaling(input_transformed_legs)
    max_euclidean_error = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM" + label + "  error_legs: ", max_euclidean_error)


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((good_model_scene_features, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((good_model_scene_features, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_image), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_image), cmap='gray')
    ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split" + label)
    ax3.imshow(np.asarray(model_image), cmap='gray')
    ax3.plot(*zip(*np.vstack((good_model_scene_features, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
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
def affine_trans_interaction_only_pose(good_model_scene_features, good_input_scene_features, model_pose, input_pose, model_image, input_image, label):
    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    #model_torso = np.vstack((model_torso, good_model_scene_features[0], good_model_scene_features[1], good_model_scene_features[10] ))
    #input_torso = np.vstack((input_torso, good_input_scene_features[0], good_input_scene_features[1], good_input_scene_features[10]))

    #model_legs = np.vstack((model_legs, good_model_scene_features[0], good_model_scene_features[1], good_model_scene_features[10] ))
    #input_legs = np.vstack((input_legs, good_input_scene_features[0], good_input_scene_features[1], good_input_scene_features[10]))


    (input_transformed_torso, M_tor) = common.find_transformation(model_torso,input_torso)
    (input_transformed_legs, M_legs) = common.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)

    max_euclidean_error_torso = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_torso: ", max_euclidean_error_torso)


    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_legs)))
    input_features_trans_norm = common.feature_scaling(input_transformed_legs)

    max_euclidean_error_legs = max_euclidean_distance(model_features_norm, input_features_trans_norm)
    print("#### AFFINE NORM " + label + "  error_legs: ", max_euclidean_error_legs)

    if max_euclidean_error_torso < 0.15 and max_euclidean_error_legs < 0.15:
        print("#### MATCH!!!  ###")
    else:
        print("#### NO MATCH!! ###")


    max_euclidean_error_torso = max_euclidean_distance(np.vstack((good_model_scene_features, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((good_model_scene_features, model_legs)), input_transformed_legs)

    print("#### AFFINE "+ label+ "  error_torso: " , max_euclidean_error_torso)
    print("#### AFFINE "+ label+ "  error_legs: ", max_euclidean_error_legs)


    markersize = 2
    ms_pose = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(np.asarray(model_image), cmap='gray')
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
             ms=ms_pose)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(np.asarray(input_image), cmap='gray')
    ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
    ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=ms_pose)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("aff trans input split " + label)
    ax3.imshow(np.asarray(model_image), cmap='gray')
    ax3.plot(*zip(*np.vstack((good_model_scene_features, model_torso, model_legs))), marker='o', color='magenta', ls='', label='model',
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

def affine_trans_interaction_pose_rand_scene(good_model_scene_features, good_input_scene_features, model_pose, input_pose,  model_image, input_image, label, plot=False):
    # TODO: Deze geeft (momenteel betere resultaten dan den _normalise => verschillen tussen matches en niet-matches is pak groter
    #TODO: normalising van whole !! en niet normaliseren van legs en torso appart
    (model_face, model_torso, model_legs) = common.split_in_face_legs_torso(model_pose)
    (input_face, input_torso, input_legs) = common.split_in_face_legs_torso(input_pose)

    # include some random features of background:
    model_torso = np.vstack((model_torso, good_model_scene_features[0], good_model_scene_features[1], good_model_scene_features[10] ))
    input_torso = np.vstack((input_torso, good_input_scene_features[0], good_input_scene_features[1], good_input_scene_features[10]))

    model_legs = np.vstack((model_legs, good_model_scene_features[0], good_model_scene_features[1], good_model_scene_features[10] ))
    input_legs = np.vstack((input_legs, good_input_scene_features[0], good_input_scene_features[1], good_input_scene_features[10]))


    (input_transformed_torso, M_tor) = common.find_transformation(model_torso, input_torso)
    (input_transformed_legs, M_legs) = common.find_transformation(model_legs, input_legs)

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]
    input_transformed_torso = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_torso))), M_tor))
    input_transformed_legs = unpad(np.dot(pad(np.vstack((good_input_scene_features, input_legs))), M_legs))

    # TODO: wanneer normaliseren? VOOR of NA berekenen van homography  ????   --> rekenenen met kommagetallen?? afrodingsfouten?
    # 1E MANIER:  NORMALISEER ALLE FEATURES = POSE + BACKGROUND
    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_torso)))
    input_features_trans_norm = common.feature_scaling(input_transformed_torso)

    euclidean_error_torso_norm = euclidean_distance(model_features_norm, input_features_trans_norm)
    #  index 2(rechts) en 5(links) zijn de polsen
    #logging.warning("Distance polsen  links: %f   rechts: %f", round(euclidean_error_torso_norm[2], 3), round(euclidean_error_torso_norm[5], 3) )
    logging.debug("#### AFFINE RAND NORM Sum torso: %f" , sum(euclidean_error_torso_norm))
    max_euclidean_error_torso_norm = max(euclidean_error_torso_norm)#max_euclidean_distance(model_features_norm, input_features_trans_norm)
    logging.debug("#### AFFINE RAND NORM " + label + "  error_torso: %f", max_euclidean_error_torso_norm)

    second_max = heapq.nlargest(2, euclidean_error_torso_norm)
    logging.debug("#### AFFINE RAND NORM 2e MAX torso: %f", second_max[1])




    model_features_norm = common.feature_scaling(np.vstack((good_model_scene_features, model_legs)))
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

    max_euclidean_error_torso = max_euclidean_distance(np.vstack((good_model_scene_features, model_torso)), input_transformed_torso)
    max_euclidean_error_legs = max_euclidean_distance(np.vstack((good_model_scene_features, model_legs)), input_transformed_legs)

    logging.debug("#### AFFINE RAND " + label + "  error_torso: %f", max_euclidean_error_torso)
    logging.debug("#### AFFINE RAND " + label + "  error_legs: %f", max_euclidean_error_legs)


    markersize = 3
    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        implot = ax1.imshow(np.asarray(model_image), cmap='gray')
        # ax1.set_title(model_image_name + ' (model)')
        ax1.set_title("model")
        ax1.plot(*zip(*good_model_scene_features), marker='o', color='magenta', ls='', label='model',
                 ms=markersize)  # ms = markersize
        ax1.plot(*zip(*model_pose), marker='o', color='blue', ls='', label='model',
                 ms=markersize)  # ms = markersize
        red_patch = mpatches.Patch(color='magenta', label='model')
        ax1.legend(handles=[red_patch])

        # ax2.set_title(input_image_name + ' (input)')
        ax2.set_title("input")
        ax2.imshow(np.asarray(input_image), cmap='gray')
        ax2.plot(*zip(*good_input_scene_features), marker='o', color='r', ls='', ms=markersize)
        ax2.plot(*zip(*input_pose), marker='o', color='blue', ls='', ms=markersize)
        ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

        ax3.set_title("aff split() " + label)
        ax3.imshow(np.asarray(model_image), cmap='gray')
        ax3.plot(*zip(*np.vstack((good_model_scene_features, model_torso, model_legs))), marker='o', color='magenta', ls='',
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
