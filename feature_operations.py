import cv2
import numpy as np
import affine_transformation
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1

def sift_detect_and_compute(image):
    # --------- SIFT FEATURE DETETCION & DESCRIPTION ------------------------
    # Initiate SIFT detector  # TODO: what is best? ORB SIFT  &&& FLANN?
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_model, des_model = sift.detectAndCompute(image,None)  # returns keypoints and descriptors
    return (kp_model, des_model)

def flann_matching(des_model, des_input, kp_model, kp_input, model_image, input_image):
    # --------- FEATURE MATCHING : FLANN MATCHER -------------------
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_model,des_input,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.80*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        model_pts = np.float32([ kp_model[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        input_pts = np.float32([ kp_input[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Find only the good, corresponding points (lines of matching points may not cross)
        M, mask = cv2.findHomography(model_pts, input_pts, cv2.RANSAC,5.0)  #tresh : 5
        matchesMask = mask.ravel().tolist()
        # TODO wat als model_image en input_image niet zelfde resolutie hebben?
        h,w = model_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        input_image_homo = cv2.polylines(input_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)  # draw homography square

        return (matchesMask, input_image_homo, good, model_pts, input_pts)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None


def plot_features(clustered_model_features, clustered_input_features, one_building, model_image, input_image):
    if one_building:  # Take the first found homography, no more computations needed
        # Reduce dimensions
        print("one building only")
        plt.figure()
        plt.scatter(clustered_model_features[:, 0], clustered_model_features[:, 1])
        # plt.scatter(model_center[:,0],model_center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Width'), plt.ylabel('Height')
        plt.imshow(model_image)
        plt.show(block=False)

        plt.figure()
        plt.scatter(clustered_input_features[:, 0], clustered_input_features[:, 1])
        # plt.scatter(input_center[:,0],input_center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Width'), plt.ylabel('Height')
        #plt.imshow(cv2.imread('img/' + input_name + '.' + img_tag))
        plt.imshow(input_image)
        plt.show(block=False)


    else: # More than one building
        plt.figure()
        for feat in clustered_model_features:
            plt.scatter(feat[:, 0], feat[:, 1], c=np.random.rand(3,), s=5)
        plt.xlabel('Height'), plt.ylabel('Weight')
        plt.imshow(model_image)
        plt.show(block=False)

        plt.figure()
        for feat in clustered_input_features:
            plt.scatter(feat[:, 0], feat[:, 1], c=np.random.rand(3,), s=5)
        plt.xlabel('Height'), plt.ylabel('Weight')
        plt.imshow(input_image)
        plt.show(block=False)


    return None


def affine_transform_urban_scene_and_pose(one_building, model_pose_features, input_pose_features, clustered_input_features, clustered_model_features,
                                          model_image, input_image):
    # -------------  CALC AFFINE TRANSFORMATION  ------------------##
    # Calc affine trans between the wrest points and some random feature points of the building
    # The question is: WHICH feature points should we take??
    # An option is to go for the "best matches" (found during featuring-matching)
    # An other option is just to take an certain number of random matches

    # Third option would be to take all the building feature points,
    # but that would probably limit transformation in aspect of the mutual spatial
    # relation between the person and the building
    # TODO: other options??

    object_index = 1  # TODO make algo that decides which object to take (the one without the person)
    # TODO: moet niet perse door een algoritme worden bepaalt, kan ook eigenschap zijn van urban scene en dus gemarkt door mens

    if not one_building:
        clustered_input_features = clustered_input_features[object_index]
        clustered_model_features = clustered_model_features[object_index]

    #### CALC AFFINE TRANSFORMATION OF WHOLE  (building feature points + person keypoints) #############"
    input_pose_features = np.append(input_pose_features, [clustered_input_features[0]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[2]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[6]], 0)
    input_pose_features = np.append(input_pose_features, [clustered_input_features[14]], 0)

    model_pose_features = np.append(model_pose_features, [clustered_model_features[0]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[2]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[6]], 0)
    model_pose_features = np.append(model_pose_features, [clustered_model_features[14]], 0)

    # Calc transformation of whole (building feature points + person keypoints)
    (input_transformed, transformation_matrix) = affine_transformation.find_transformation(model_pose_features,input_pose_features)

    #####################################################################################


    ### CALC FIRST AFFINE TrANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    # Some random selected features (of the buidling)
    #input_building_features = np.array([clustered_input_features[0], clustered_input_features[2], clustered_input_features[6], clustered_input_features[14]])
    #model_building_features = np.array([clustered_model_features[0], clustered_model_features[2], clustered_model_features[6], clustered_model_features[14]])

    # Calc the transformation matrix
    #(input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(model_building_features,input_building_features)

    # Calc the transformed features again with same transformation matrix
    # But now with the people keypoints appended

    #pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #unpad = lambda x: x[:, :-1]
    #transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #input_transformed = transform(input_pose_features)



    # ------------------ 2 BUILDINGS --------------------------

    #     ### EERSTE MANIER: calc affine trans of whole (building features + person key-points)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(whole_output_features,
    #                                                                                            whole_input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### TWEEDE MANIER: CALC FIRST AFFINE TANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     # input_building_features = np.array(
    #     #     [clustered_input_features[object_index][0], clustered_input_features[object_index][2], clustered_input_features[object_index][6],
    #     #      clustered_input_features[object_index][14]])
    #     # model_building_features = np.array(
    #     #     [clustered_model_features[object_index][0], clustered_model_features[object_index][2], clustered_model_features[object_index][6],
    #     #      clustered_model_features[object_index][14]])
    #     #
    #     # # Calc the transformation matrix of building features
    #     # (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #     #     model_building_features,
    #     #     input_building_features)
    #
    #
    #
    #     #3e MANIER:  Calc transformation of person-features
    #     input_features = np.append(input_features, clustered_input_features[object_index], 0)
    #     output_features = np.append(output_features, clustered_model_features[object_index], 0)
    #
    #     #print("ejejeje: " , input_features)
    #
    #
    #
    #     (input_person_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #         output_features,
    #         input_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #     img = cv2.imread('img/' + model_name + '.' + img_tag)
    #     rows, cols, ch = img.shape
    #
    #     input_features = input_features.astype(np.float32)
    #     output_features = output_features.astype(np.float32)
    #     pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    #     pts2 = output_features #np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    #
    #     print("pts: ", pts1)
    #
    #
    #     print("whoooole" , input_features)
    #     '''
    #     print("whoooole", output_features)
    #     print("shape: " , input_features.shape)
    #     print("shape2; " , output_features.shape)
    #     '''

    markersize = 3

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    # ax1.set_title(model_image_name + ' (model)')
    ax1.set_title("model")
    ax1.plot(*zip(*model_pose_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    red_patch = mpatches.Patch(color='magenta', label='model')
    ax1.legend(handles=[red_patch])

    # ax2.set_title(input_image_name + ' (input)')
    ax2.set_title("input")
    ax2.imshow(input_image)
    ax2.plot(*zip(*input_pose_features), marker='o', color='r', ls='', ms=markersize)
    ax2.legend(handles=[mpatches.Patch(color='red', label='input')])

    ax3.set_title("transformation")
    ax3.imshow(model_image)
    ax3.plot(*zip(*model_pose_features), marker='o', color='magenta', ls='', label='model',
             ms=markersize)  # ms = markersize
    ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
                        mpatches.Patch(color='magenta', label='model')])
    #plt.tight_layout()
    plt.show(block=False)
    return None

    # cv2.waitKey(0)

    # if one_building: # Take the first found homography, no more computations needed
    #     #Reduce dimensions
    #     clustered_model_features = np.squeeze(clustered_features[0])
    #     clustered_input_features = np.squeeze(clustered_features[1])
    #     print("one building only")
    #
    #     plt.scatter(clustered_model_features[:,0],clustered_model_features[:,1])
    #     #plt.scatter(model_center[:,0],model_center[:,1],s = 80,c = 'y', marker = 's')
    #     plt.xlabel('Width'),plt.ylabel('Height')
    #     plt.imshow(model_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     plt.scatter(clustered_input_features[:,0],clustered_input_features[:,1])
    #     #plt.scatter(input_center[:,0],input_center[:,1],s = 80,c = 'y', marker = 's')
    #     plt.xlabel('Width'),plt.ylabel('Height')
    #     plt.imshow(cv2.imread('img/' + input_name + '.' + img_tag))
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     #-------------  CALC AFFINE TRANSFORMATION  ------------------##
    #     # Calc affine trans between the wrest points and some random feature points of the building
    #     # The question is: WHICH feature points should we take??
    #     # An option is to go for the "best matches" (found during featuring-matching)
    #     # An other option is just to take an certain number of random matches
    #
    #     # Third option would be to take all the building feature points,
    #     # but that would probably limit transformation in aspect of the mutual spatial
    #     # relation between the person and the building
    #     # TODO: other options??
    #
    #     # Create feature array for first person
    #     # feature points of pisa tower are in A
    #     # feautes van pols =
    #
    #     # p9_r_pols = np.array([[152, 334]])  #pisa9
    #     # p9_l_pols = np.array([[153,425]])
    #     # p10_r_pols = np.array([[256, 362]])   #pisa10
    #     # p10_l_pols = np.array([[247, 400]])
    #
    #     input_features = np.array([[152, 334], [153, 425]])  #pisa9
    #     output_features = np.array([[256, 362], [247, 400]]) #pisa10
    #
    #
    #
    #     input_features =  np.array([[391,92]])  #taj3  enkel recher pols
    #     input_features = np.array([[463, 89]]) # foute locatie
    #     input_features = np.array([[391, 92], [517, 148]])  # taj3  enkel recher pols + nek
    #     input_features = np.array([[391, 92], [429, 126]])  # taj3  enkel recher pols + r elbow
    #
    #     output_features = np.array([[303,37]]) #taj4 enkel rechter pols
    #     output_features = np.array([[303, 37],[412, 90]])  # taj4 enkel rechter pols + nek
    #     output_features = np.array([[303, 37], [347, 70]])  # taj4 enkel rechter pols + r elbow
    #
    #     #### CALC AFFINE TRANSFORMATION OF WHOLE  (building feature points + person keypoints) #############"
    #
    #     input_features = np.append(input_features, [clustered_input_features[0]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[2]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[6]], 0)
    #     input_features = np.append(input_features, [clustered_input_features[14]], 0)
    #
    #     output_features = np.append(output_features, [clustered_model_features[0]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[2]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[6]], 0)
    #     output_features = np.append(output_features, [clustered_model_features[14]], 0)
    #
    #
    #     # Calc transformation of whole (building feature points + person keypoints)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(output_features,
    #                                                                                            input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### CALC FIRST AFFINE TrANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     input_building_features = np.array([clustered_input_features[0], clustered_input_features[2], clustered_input_features[6], clustered_input_features[14]])
    #     model_building_features = np.array([clustered_model_features[0], clustered_model_features[2], clustered_model_features[6], clustered_model_features[14]])
    #
    #
    #
    #     # Calc the transformation matrix
    #     (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(model_building_features,
    #                                                                                            input_building_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #
    #
    #
    #     markersize = 3
    #
    #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    #     implot = ax1.imshow(model_image)
    #     # ax1.set_title(model_image_name + ' (model)')
    #     ax1.set_title("model")
    #     ax1.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     red_patch = mpatches.Patch(color='magenta', label='model')
    #     ax1.legend(handles=[red_patch])
    #
    #     # ax2.set_title(input_image_name + ' (input)')
    #     ax2.set_title("input")
    #     ax2.imshow(input_image)
    #     ax2.plot(*zip(*input_features), marker='o', color='r', ls='', ms=markersize)
    #     ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
    #
    #     ax3.set_title("transformation")
    #     ax3.imshow(model_image)
    #     ax3.plot(*zip(*output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    #     ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
    #                         mpatches.Patch(color='magenta', label='model')])
    #     #plt.tight_layout()
    #     plt.show(block=False)
    #
    #     plt.figure()
    #
    #
    #
    #
    #
    # else: # More than one building
    #     for feat in clustered_model_features:
    #         plt.scatter(feat[:, 0], feat[:, 1])
    #     plt.xlabel('Height'),plt.ylabel('Weight')
    #     plt.imshow(model_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #
    #     for feat in clustered_input_features:
    #         plt.scatter(feat[:, 0], feat[:, 1])
    #     plt.xlabel('Height'),plt.ylabel('Weight')
    #     plt.imshow(input_image)
    #     plt.show(block=False)
    #     plt.figure()
    #
    #     # -------------  CALC AFFINE TRANSFORMATION  ------------------##
    #
    #     # p9_r_pols = np.array([[152, 334]])  #pisa9
    #     # p9_l_pols = np.array([[153,425]])
    #     # p10_r_pols = np.array([[256, 362]])   #pisa10
    #     # p10_l_pols = np.array([[247, 400]])
    #
    #     input_features = np.array([[256, 362], [247, 400]], np.float32)   # pisa9
    #     output_features = np.array([[152, 334], [153, 425]], np.float32) # pisa10
    #
    #
    #     input_features = np.array([[127, 237], [206, 234], [317, 205] ], np.float32)  # trap1
    #     #input_features = np.array([[218, 299], [280, 300]])  # trap3
    #     #input_features = np.array([[136, 230], [297, 536], [343, 542]])  #trap9  rpols, renkel, lenkel
    #     input_features = np.array([[113, 290], [179, 290]], np.float32)  # trap1
    #
    #     output_features = np.array([[116, 289], [188, 284], [307, 257]], np.float32)  # trap2
    #     #output_features = np.array([[150, 230],[319, 570], [376, 587]], np.float32) #trap8   rpols, renkel, lenkel
    #     #output_features = np.array([[254, 248], [293, 253]], np.float32)  # trap4
    #     output_features = np.array([[127, 237], [206, 234]], np.float32)  # trap1
    #
    #     object_index = 1
    #
    #     whole_input_features = np.append(input_features, [clustered_input_features[object_index][0]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][2]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][6]], 0)
    #     whole_input_features = np.append(whole_input_features, [clustered_input_features[object_index][8]], 0)
    #
    #     whole_output_features = np.append(output_features, [clustered_model_features[object_index][0]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][2]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][6]], 0)
    #     whole_output_features = np.append(whole_output_features, [clustered_model_features[object_index][8]], 0)
    #
    #     ### EERSTE MANIER: calc affine trans of whole (building features + person key-points)
    #     (input_transformed, transformation_matrix) = affine_transformation.find_transformation(whole_output_features,
    #                                                                                            whole_input_features)
    #
    #     #####################################################################################
    #
    #
    #     ### TWEEDE MANIER: CALC FIRST AFFINE TANS MATRIX OF ONLY THE BUILDING FEATURES ###################"
    #
    #     # Some random selected features (of the buidling)
    #     # input_building_features = np.array(
    #     #     [clustered_input_features[object_index][0], clustered_input_features[object_index][2], clustered_input_features[object_index][6],
    #     #      clustered_input_features[object_index][14]])
    #     # model_building_features = np.array(
    #     #     [clustered_model_features[object_index][0], clustered_model_features[object_index][2], clustered_model_features[object_index][6],
    #     #      clustered_model_features[object_index][14]])
    #     #
    #     # # Calc the transformation matrix of building features
    #     # (input_building_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #     #     model_building_features,
    #     #     input_building_features)
    #
    #
    #
    #     #3e MANIER:  Calc transformation of person-features
    #     input_features = np.append(input_features, clustered_input_features[object_index], 0)
    #     output_features = np.append(output_features, clustered_model_features[object_index], 0)
    #
    #     #print("ejejeje: " , input_features)
    #
    #
    #
    #     (input_person_transformed, transformation_matrix) = affine_transformation.find_transformation(
    #         output_features,
    #         input_features)
    #
    #     # Calc the transformed features again with same transformation matrix
    #     # But now with the people keypoints appended
    #     pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    #     unpad = lambda x: x[:, :-1]
    #     transform = lambda x: unpad(np.dot(pad(x), transformation_matrix))
    #     input_transformed = transform(input_features)
    #
    #     img = cv2.imread('img/' + model_name + '.' + img_tag)
    #     rows, cols, ch = img.shape
    #
    #     input_features = input_features.astype(np.float32)
    #     output_features = output_features.astype(np.float32)
    #     pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    #     pts2 = output_features #np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    #
    #     print("pts: ", pts1)
    #
    #
    #     print("whoooole" , input_features)
    #     '''
    #     print("whoooole", output_features)
    #     print("shape: " , input_features.shape)
    #     print("shape2; " , output_features.shape)
    #     '''
    #
    #
    #
    #
    #     markersize = 3
    #
    #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    #     implot = ax1.imshow(model_image)
    #     # ax1.set_title(model_image_name + ' (model)')
    #     ax1.set_title("model")
    #     ax1.plot(*zip(*whole_output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     red_patch = mpatches.Patch(color='magenta', label='model')
    #     ax1.legend(handles=[red_patch])
    #
    #     # ax2.set_title(input_image_name + ' (input)')
    #     ax2.set_title("input")
    #     ax2.imshow(input_image)
    #     ax2.plot(*zip(*whole_input_features), marker='o', color='r', ls='', ms=markersize)
    #     ax2.legend(handles=[mpatches.Patch(color='red', label='input')])
    #
    #     ax3.set_title("transformation")
    #     ax3.imshow(model_image)
    #     ax3.plot(*zip(*whole_output_features), marker='o', color='magenta', ls='', label='model',
    #              ms=markersize)  # ms = markersize
    #     ax3.plot(*zip(*input_transformed), marker='o', color='b', ls='', ms=markersize)
    #     ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'),
    #                         mpatches.Patch(color='magenta', label='model')])
    #     # plt.tight_layout()
    #     plt.show(block=False)
    #
    #     plt.figure()
    #
    #
    #
    # plt.show()
    # cv2.waitKey(0)

