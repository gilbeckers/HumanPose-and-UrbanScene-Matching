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
def rename_files():
    print("not implemented")

def replace_json_files(pose):
    for foto in glob.iglob(poses+pose+"/fotosfout/*"):
        foto = foto.split(".")[0];
        foto = foto.replace("fotosfout","json")
        foto = foto +".json"
        os.system("mv "+foto+" "+poses+pose+"/jsonfout/  2>/dev/null")

#**************************************precision recall********************************************
def calculate_pr(pose,tresh):
    path = poses+pose
    model = path+"/json/"+pose+".json"
    #pose = "1"
    #path = '/media/jochen/2FCA69D53AB1BFF41/dataset/poses/pose'+pose
    #model = path+"/json/0.json"
    model_pose_features = common.parse_JSON_multi_person(model)
    feature_name = 'orb-flann'
    detector, matcher = features.init_feature(feature_name)
    model_image = cv2.imread(model.split(".")[0].replace("json","fotos")+".jpg", cv2.IMREAD_GRAYSCALE)
    print(model.split(".")[0].replace("json","fotos")+".jpg")
    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        input_pose_features= common.parse_JSON_multi_person(json)
        input_image = cv2.imread(json.split(".")[0].replace("json","fotos")+".jpg", cv2.IMREAD_GRAYSCALE)
        result_whole = matching.match_whole(model_pose_features, input_pose_features, detector, matcher, model_image, input_image,False, False)
        if result_whole < tresh:
            true_positive = true_positive +1
        else:
            false_negative = false_negative +1

    for json in glob.iglob(path+"/jsonfout/*.json"):
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
