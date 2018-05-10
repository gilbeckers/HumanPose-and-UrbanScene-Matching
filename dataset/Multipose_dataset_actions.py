import glob
import logging
import os
import sys
import cv2
import common
from urbanscene import features
import matching
import common
import posematching.multi_person as multiperson
import posematching.multi_person
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("Multipose dataset")

schijfnaam = "/media/jochen/2FCA69D53AB1BFF43/"
poses = schijfnaam+'dataset/kever/poses/'
json = schijfnaam+'dataset/kever/json/'
fotos = schijfnaam+'dataset/kever/fotos/'

#*****************************************logic*********************************************
def replace_json_files():
    pose = "14"
    galabal = galabal_18
    path = galabal+pose
    for foto in glob.iglob(path+"/fotosfout/*"):
        foto = foto.split(".")[0];
        foto = foto.replace("fotosfout","json")
        foto = foto +".json"
        os.system("mv "+foto+" "+path+"/jsonfout/ ")



#*****************************************logic*********************************************
def rename_files():
    print("not implemented")




#**************************************precision recall********************************************
def calculate_pr(pose,tresh):
    path = poses+pose
    model = path+"/model.json" #take filtered model for keypoints
    #pose = "1"
    #path = '/media/jochen/2FCA69D53AB1BFF41/dataset/poses/pose'+pose
    #model = path+"/json/0.json"
    model_pose_features = common.parse_JSON_multi_person(model)
    model = path+"/json/"+pose+".json"
    feature_name = 'orb-flann'
    detector, matcher = features.init_feature(feature_name)
    model_image = cv2.imread(model.split(".")[0].replace("json","fotos")+".jpg", cv2.IMREAD_GRAYSCALE)
    print(model.split(".")[0].replace("json","fotos")+".jpg")
    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        print(json)
        input_pose_features= common.parse_JSON_multi_person(json)
        input_image = cv2.imread(json.split(".")[0].replace("json","fotos")+".jpg", cv2.IMREAD_GRAYSCALE)
        result_whole = matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image,False, False)
        if result_whole < tresh:
            true_positive = true_positive +1
        else:
            false_negative = false_negative +1

    for json in glob.iglob(path+"/jsonfout/*.json"):
        print(json)
        input_pose_features= common.parse_JSON_multi_person(json)
        input_image = cv2.imread(json.split(".")[0].replace("json","fotos")+".jpg", cv2.IMREAD_GRAYSCALE)
        result_whole = matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image,False, False)
        if result_whole < tresh:
            false_positive = false_positive +1
        else:
            true_negative = true_negative +1

    precision = 0
    recall =0
    if (true_positive+false_positive) !=0:
        precision = true_positive / (true_positive+false_positive)
    if  (true_positive+false_negative) !=0:
        recall = true_positive / (true_positive+false_negative)

    #******** print small raport ******************

    print ("*************raport of pose "+pose+" ****************")
    print ("true_positive: " + str(true_positive))
    print ("false_positive: "+ str(false_positive))
    print ("true_negative: " + str(true_negative))
    print ("false_negative: "+ str(false_negative))
    print ("recall: "+ str(recall))
    print ("precision: "+ str(precision))
    print ("error_score_tresh: "+str(error_score_tresh))
    return (precision,recall)


def make_pr_curve(pose):
    precisions = []
    recalls = []
    start_tresh = 0
    for i in range(0,10):
        tresh = start_tresh + i*0.1
        (precision,recall) = calculate_pr(pose,tresh)
        precisions.append(precision)
        recalls.append(recall)

    return(precisions,recalls)


def draw_pr_curve():
    pose = "29"
    precisions, recalls = make_pr_curve(pose)

    plt.plot(recalls,precisions, label="describing poses")

    plt.legend(loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('Pr curves of urban algorithm')
    plt.axis([0,1,0,1])
    plt.legend()

    plt.show()



#*************************************accuracy*************************************
def calculate_accuracy(pose, path,error_score_tresh,superimp):

    model = path+"/json/"+pose+".json"

    model_features = common.parse_JSON_multi_person(model)

    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True, superimp)
        if error_score < error_score_tresh:
            true_positive = true_positive +1

        else:
            false_negative = false_negative +1


    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True, superimp)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
        else:
            true_negative = true_negative +1


    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = 0
    recall = 0
    if (true_positive+false_positive) !=0:
        precision = true_positive / (true_positive+false_positive)
    if  (true_positive+false_negative) !=0:
        recall = true_positive / (true_positive+false_negative)
    return accuracy, precision, recall

def find_best_accuracy():
    pose = "00100"
    path = multipose+pose
    error_tresh_start = 0
    global error_tresh
    superimp = False
    best_accuracy = 0
    best_tresh = 0
    best_recall =0
    best_precision =0
    for i in range(0,400):
        error_tresh = error_tresh_start + 0.005*i
        (accuracy, precision, recall)= calculate_accuracy(pose, path, error_tresh, superimp)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_tresh = error_tresh
            best_recall = recall
            best_precision = precision

    print ("*************raport best accuray****************")
    print ("best_accuracy: " + str(best_accuracy))
    print ("best_tresh: "+ str(best_tresh))
    print ("recall: "+ str(recall))
    print ("precision: "+ str(precision))


#************************************find special in pr******************************
def find_specials():
    pose = "00100"
    path = multipose+pose # galabal_economica+pose #multipose+pose #galabal_18+pose  #
    error_tresh_start = 1
    global error_tresh
    prev_pres =1
    prev_rec =0

    # for i in range(0,400):
    #     error_tresh = error_tresh_start + 0.005*i
    #     (precision,recall) = calculate_pr(pose,path,error_tresh,False)
    #     precisions.append(precision)
    #     recalls.append(recall)
    #
    # for i in range(0,400):
    #     error_tresh = error_tresh_start + 0.005*i
    #     (precision,recall) = calculate_pr(pose,path,error_tresh,True)
    #     precisions2.append(precision)
    #     recalls2.append(recall)

    for i in range(0,5):
        error_tresh = error_tresh_start + 0.1*i
        if prev_rec >0.58 and prev_rec <0.65:
            (precision,recall) = find_specials_loop(pose,path,error_tresh,True, True)
            prev_rec = recall
            prev_pres = precision
        else:
            (precision,recall) = find_specials_loop(pose,path,error_tresh,False, False)
            prev_rec = recall
            prev_pres = precision



def find_specials_loop(pose,path,error_score_tresh, print_false_pos, print_false_neg):


    model = path+"/json/"+pose+".json"

    model_features = common.parse_JSON_multi_person(model)
    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, True)
        if error_score < error_score_tresh:
            true_positive = true_positive +1

        else:
            false_negative = false_negative +1
            if print_false_neg:
                print("false pos at: "+json)


    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            if print_false_pos:
                print("false pos at: "+json)
        else:
            true_negative = true_negative +1

    precision = 0
    recall =0
    if (true_positive+false_positive) !=0:
        precision = true_positive / (true_positive+false_positive)
    if  (true_positive+false_negative) !=0:
        recall = true_positive / (true_positive+false_negative)

    #******** print small raport ******************

    print ("*************raport of pose "+pose+" ****************")
    print ("true_positive: " + str(true_positive))
    print ("false_positive: "+ str(false_positive))
    print ("true_negative: " + str(true_negative))
    print ("false_negative: "+ str(false_negative))
    print ("recall: "+ str(recall))
    print ("precision: "+ str(precision))
    print ("error_score_tresh: "+str(error_score_tresh))
    return (precision,recall)
