import glob
import logging
import os

import common
import posematching.multi_person as multiperson
import posematching.multi_person
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
logger = logging.getLogger("Multipose dataset")
schijfnaam = '/media/jochen/2FCA69D53AB1BFF42'
multipose = schijfnaam+'dataset/Multipose/fotos/'
poses = schijfnaam+'/dataset/Multipose/poses/'
data = schijfnaam+'/dataset/Multipose/json/'
poses = schijfnaam+'/dataset/Multipose/poses/'

galabal_18 = schijfnaam+'/dataset/galabal2018/poses/'
galabal_18_json = schijfnaam+'/dataset/galabal2018/json/'
galabal_18_fotos = schijfnaam+'/dataset/galabal2018/fotos/'

galabal_17 = schijfnaam+'/dataset/galabal2017/poses/'
galabal_17_json = schijfnaam+'/dataset/galabal2017/json/'
galabal_17_fotos = schijfnaam+'/dataset/galabal2017/fotos/'

galabal_economica = schijfnaam+'/dataset/galabal_Ekonomica/poses/'
galabal_economica_json = schijfnaam+'/dataset/galabal_Ekonomica/json/'
galabal_economica_fotos = schijfnaam+'/dataset/galabal_Ekonomica/fotos/'

galabal_medica = schijfnaam+'/dataset/galabal_Medica/poses/'
galabal_medica_json = schijfnaam+'/dataset/galabal_Medica/json/'
galabal_medica_fotos = schijfnaam+'/dataset/galabal_Medica/fotos/'

galabal_medica2 = schijfnaam+'/dataset/galabal_Medica2/poses/'
galabal_medica2_json = schijfnaam+'/dataset/galabal_Medica2/json/'
galabal_medica2_fotos = schijfnaam+'/dataset/galabal_Medica2/fotos/'

galabal_psycho = schijfnaam+'/dataset/galabal_psycho/poses/'
galabal_psycho_json = schijfnaam+'/dataset/galabal_psycho/json/'
galabal_psycho_fotos = schijfnaam+'/dataset/galabal_psycho/fotos/'

#pose should look like 00100
def find_matches_with(pose):

    global eucl_dis_tresh_torso
    global rotation_tresh_torso
    global eucl_dis_tresh_legs
    global rotation_tresh_legs
    global eucld_dis_shoulders_tresh

    eucl_dis_tresh_torso= 0.18
    rotation_tresh_torso= 19
    eucl_dis_tresh_legs= 0.058
    rotation_tresh_legs= 24
    eucld_dis_shoulders_tresh= 0.125
    if len(pose) == 5 and pose.isdigit():
        model = data+pose+"_keypoints.json"
        model_features = common.parse_JSON_multi_person(model)
        count = 0
        os.system("mkdir -p "+poses+pose)
        os.system("mkdir -p "+poses+pose+"/json")
        os.system("mkdir -p "+poses+pose+"/jsonfout")
        os.system("mkdir -p "+poses+pose+"/fotos")
        os.system("mkdir -p "+poses+pose+"/fotosfout")
        for json in glob.iglob(data+"*_keypoints.json"):
            logger.info(json)
            input_features = common.parse_JSON_multi_person(json)
            (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
            if result == True:
                place = json.split("_keypoints")[0]
                place = place.split("json/")[1]
                place = place+".json"
                os.system("cp "+json+" "+poses+pose+"/json/"+place)
                foto = json.split("_keypoints")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".jpg"
                os.system("cp "+foto+" "+poses+pose+"/fotos/")
                count = count+1
                logger.info("true")
        print ("there are "+str(count)+" matches found")

    else:
        print ("find_matches_with has wrong input")



def check_matches(pose):
    global eucl_dis_tresh_torso
    global rotation_tresh_torso
    global eucl_dis_tresh_legs
    global rotation_tresh_legs
    global eucld_dis_shoulders_tresh

    eucl_dis_tresh_torso= 0.18
    rotation_tresh_torso= 19
    eucl_dis_tresh_legs= 0.058
    rotation_tresh_legs= 24
    eucld_dis_shoulders_tresh= 0.125
    path = poses+pose
    model = path+"/json/"+pose+".json"

    model_features = common.parse_JSON_multi_person(model)

    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if result == True:
            true_positive = true_positive +1
        else:
            false_negative = false_negative +1
            print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if result == True:
            false_positive = false_positive +1
            print ("error at: "+json)
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


#*****************************************logic*********************************************
def rename_files():
    i=0
    foto_path = galabal_17_fotos
    json_path = galabal_17_json
    for json in glob.iglob(json_path+"*_keypoints.json"):
        i = i+1
        os.system("mv "+json+" "+json_path+str(i)+".json")
        foto = json.split("_keypoints")[0];
        foto = foto.replace("json","fotos")
        foto = foto +".jpg"
        os.system("mv "+foto+" "+foto_path+str(i)+".jpg")

def replace_json_files():
    pose = "14"
    galabal = galabal_18
    path = galabal+pose
    for foto in glob.iglob(path+"/fotosfout/*"):
        foto = foto.split(".")[0];
        foto = foto.replace("fotosfout","json")
        foto = foto +".json"
        os.system("mv "+foto+" "+path+"/jsonfout/ ")

#***********************************galabal dataset_actions*********************************
def find_galabal_matches():
    pose = "7"
    galabal = galabal_medica
    galabaljson = galabal_medica_json

    model = galabaljson+pose+".json"
    model_features = common.parse_JSON_multi_person(model)
    count = 0
    os.system("mkdir -p "+galabal+pose)
    os.system("mkdir -p "+galabal+pose+"/json")
    os.system("mkdir -p "+galabal+pose+"/jsonfout")
    os.system("mkdir -p "+galabal+pose+"/fotos")
    os.system("mkdir -p "+galabal+pose+"/fotosfout")
    for json in glob.iglob(galabaljson+"*.json"):
        logger.info(json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
        if result == True:
            place = json.split(".json")[0]
            place = place.split("json/")[1]
            place = place+".json"
            os.system("cp "+json+" "+galabal+pose+"/json/"+place)
            foto = json.split(".json")[0];
            foto = foto.replace("json","fotos")
            foto = foto +".jpg"
            os.system("cp "+foto+" "+galabal+pose+"/fotos/")
            count = count+1
            logger.info("true")
    print ("there are "+str(count)+" matches found")

def check_galabal_matches(pose):
    galabal = galabal_18
    model = galabal+pose+"/json/"+pose+".json"
    model_features = common.parse_JSON_multi_person(model)
    count =0
    for json in glob.iglob(galabal+pose+"/json/*.json"):

        logger.info(json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
        if result == False:
            count = count +1
            print ("error at: "+json)
    print (str(count)+" false negatives")
    count = 0
    for json in glob.iglob(galabal+pose+"/jsonfout/*.json"):
        logger.info(json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
        if result == True:
            count = count +1
            #print ("error at: "+json)
    print (str(count)+" false positves")



#****************************************test_script**********************
def test_script():
    pose = "14"
    galabal = galabal_18
    galabaljson = galabal_18_json

    model = galabaljson+pose+".json"
    model_features = common.parse_JSON_multi_person(model)

    input = galabaljson+"16.json"
    input_features = common.parse_JSON_multi_person(input)

    (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
    print (result)
    print (error_score)

#**************************************precision recall********************************************
def calculate_pr(pose,error_score_tresh):

    path = poses+pose
    model = path+"/json/"+pose+".json"

    model_features = common.parse_JSON_multi_person(model)

    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            true_positive = true_positive +1

        else:
            false_negative = false_negative +1


    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
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

    precisions = [];
    recalls = []
    error_tresh_start = 0
    for i in range(0,100):
        error_tresh = error_tresh_start + 0.05*i
        (precision,recall) = calculate_pr(pose,error_tresh)
        precisions.append(precision)
        recalls.append(recall)

    return (precisions,recalls)


def draw_pr_curve(pose):
    (precisions,recalls) = make_pr_curve(pose)
    plt.plot(recalls,precisions)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('Pr curve of pose : '+pose)
    plt.axis([0,1,0,1])
    plt.legend()

    plt.show()
