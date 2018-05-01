import glob
import logging
import os

import common
import posematching.multi_person as multiperson
import posematching.multi_person
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
logger = logging.getLogger("Multipose dataset")
schijfnaam = '/media/jochen/2FCA69D53AB1BFF43'

multipose = schijfnaam+'/dataset/Multipose/poses/'
multipose_json = schijfnaam+'/dataset/Multipose/json/'
multipose_fotos = schijfnaam+'/dataset/Multipose/fotos/'

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

heart = schijfnaam+'/dataset/heart/poses/'
heart_json = schijfnaam+'/dataset/heart/json/'
heart_fotos = schijfnaam+'/dataset/heart/fotos/'

def test_script():
    pose = "00100"
    galabal = multipose
    galabaljson = multipose_json

    model = galabaljson+pose+"_keypoints.json"
    model_features = common.parse_JSON_multi_person(model)

    input = galabaljson+"00031_keypoints.json"
    input_features = common.parse_JSON_multi_person(input)

    (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, normalise=True)
    print (result)
    print (error_score)

#pose should look like 00100
def find_matches_with(pose):

    if len(pose) == 5 and pose.isdigit():
        model = multipose_json+pose+"_keypoints.json"
        model_features = common.parse_JSON_multi_person(model)
        count = 0
        os.system("mkdir -p "+multipose+pose)
        os.system("mkdir -p "+multipose+pose+"/json2")
        os.system("mkdir -p "+multipose+pose+"/jsonfout")
        os.system("mkdir -p "+multipose+pose+"/fotos2")
        os.system("mkdir -p "+multipose+pose+"/fotosfout")
        for json in glob.iglob(multipose_json+"*_keypoints.json"):
            input_features = common.parse_JSON_multi_person(json)
            (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, normalise=True)
            if result == True:
                place = json.split("_keypoints")[0]
                place = place.split("json/")[1]
                place = place+".json"
                os.system("cp "+json+" "+multipose+pose+"/json2/"+place)
                foto = json.split("_keypoints")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".jpg"
                os.system("cp "+foto+" "+multipose+pose+"/fotos2/")
                count = count+1
                logger.info("true")
        print ("there are "+str(count)+" matches found")

    else:
        print ("find_matches_with has wrong input")



def check_matches(pose):

    path = multipose+pose
    model = path+"/json/"+pose+".json"

    model_features = common.parse_JSON_multi_person(model)

    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, True)
        if result == True:
            true_positive = true_positive +1
        else:
            false_negative = false_negative +1
            #print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, True)
        if result == True:
            false_positive = false_positive +1
            #print ("error at: "+json)
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
    foto_path = heart_fotos
    json_path = heart_json
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
    galabal = galabal_medica2
    galabaljson = galabal_medica2_json

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
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, normalise=True)
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

def check_galabal_matches():
    pose = "14"
    galabal = galabal_18
    model = galabal+pose+"/json/"+pose+".json"
    model_features = common.parse_JSON_multi_person(model)
    count =0
    for json in glob.iglob(galabal+pose+"/json/*.json"):

        logger.info(json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
        if error_score > 0.8:
            count = count +1
            # foto = json.split(".json")[0];
            # foto = foto.replace("json","fotos")
            # foto = foto +".jpg"
            # os.system("mv "+foto+" "+galabal+pose+"/fotos/")
            print ("error at: "+json)
    print (str(count)+" false negatives")
    count = 0
    for json in glob.iglob(galabal+pose+"/jsonfout/*.json"):
        logger.info(json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
        if error_score < 0.2:
            count = count +1
            print ("error at: "+json)
    #replace_json_files()
    print (str(count)+" false positves")



#****************************************test_script**********************
def test_script_galabal():
    pose = "14"
    galabal = galabal_18
    galabaljson = galabal_18_json

    model = galabaljson+pose+".json"
    model_features = common.parse_JSON_multi_person(model)

    input = galabaljson+"229.json"
    input_features = common.parse_JSON_multi_person(input)

    (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
    print (result)
    print (error_score)

#**************************************precision recall********************************************
def calculate_pr(pose,path,error_score_tresh,superimp):


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




def calculate_pr_2(pose,path,error_score_tresh):


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


    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match2(model_features, input_features, True)
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


def make_pr_curve(pose, path):

    precisions = [];
    recalls = []
    precisions2 = [];
    recalls2 = []
    precisions3 = [];
    recalls3 = []
    error_tresh_start = 0
    global error_tresh

    for i in range(0,400):
        error_tresh = error_tresh_start + 0.005*i
        (precision,recall) = calculate_pr(pose,path,error_tresh,False)
        precisions.append(precision)
        recalls.append(recall)

    for i in range(0,400):
        error_tresh = error_tresh_start + 0.005*i
        (precision,recall) = calculate_pr(pose,path,error_tresh,True)
        precisions2.append(precision)
        recalls2.append(recall)

    for i in range(0,400):
        error_tresh = error_tresh_start + 0.01*i
        (precision,recall) = calculate_pr_2(pose,path,error_tresh)
        precisions3.append(precision)
        recalls3.append(recall)

    return (precisions,recalls,precisions2,recalls2,precisions3,recalls3)

def draw_pr_curve():
    pose = "00100"
    path = multipose+pose # galabal_economica+pose #multipose+pose #galabal_18+pose  #

    (precisions,recalls,precisions2,recalls2,precisions3,recalls3) = make_pr_curve(pose,path)
    plt.plot(recalls,precisions, label="Affine transformation without pose overlapping")
    plt.plot(recalls2,precisions2, label="Affine transformation with pose overlapping")
    plt.plot(recalls3,precisions3, label="describing poses")



    plt.legend(loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('Pr curves of Multipose algorithms')
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
