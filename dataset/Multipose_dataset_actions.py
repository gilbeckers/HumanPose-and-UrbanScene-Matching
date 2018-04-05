import glob
import logging
import os

import common
import posematching.multi_person as multiperson
import posematching.calcAngle as calcAngle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
logger = logging.getLogger("Multipose dataset")

Singleposes = '/media/jochen/2FCA69D53AB1BFF41/dataset/poses/pose'
multipose = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/fotos/'
poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/poses/'
data = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/json/'
poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/poses/'

galabal = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/poses/'
galabaljson = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/json/'
galabalfotos = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/fotos/'



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

def test_script():

    model = data+"00100_keypoints.json"
    model_features = common.parse_JSON_multi_person(model)

    input = data+"00100_keypoints.json"
    input_features = common.parse_JSON_multi_person(input)
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

    input_features = common.parse_JSON_multi_person(input)
    (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, normalise=True)
    print (result)
    print (error_score)

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
    recall = 0
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




#***********************************galabal dataset_actions*********************************
def find_galabal_matches(pose):
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
    print (str(count)+" foto's werden niet meer herkend")

#*****************************************logic*********************************************
def rename_files():
    i=0
    for json in glob.iglob(galabal+"*_keypoints.json"):
        i = i+1
        os.system("cp "+json+" "+galabal+str(i)+".json")
        foto = json.split("_keypoints")[0];
        foto = foto.replace("json","fotos")
        foto = foto +".jpg"
        os.system("cp "+foto+" "+galabalfotos+str(i)+".jpg")

def replace_json_files(pose):
    for foto in glob.iglob(poses+pose+"/fotos/*"):
        print(foto)
        foto = foto.split(".")[0];
        foto = foto.replace("fotos","json")
        foto = foto +".json"
        os.system("mv "+foto+" "+poses+pose+"/json/  2>/dev/null")

#**************************************precision recall********************************************
def calculate_pr(pose,error_score_tresh):
    '''
    path = poses+pose
    model = path+"/json/"+pose+".json"
    '''
    poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/poses/pose'
    pose = "1"
    path = poses+pose
    model = path+"/json/0.json"
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
            #print ("score is: "+str(error_score))
        else:
            false_negative = false_negative +1
            #print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "2"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "3"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "4"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "5"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "6"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

#//////////////////////////////////////////////check for pose5
    pose = "2"
    path = poses+pose
    model = path+"/json/0.json"
    model_features = common.parse_JSON_multi_person(model)
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            true_positive = true_positive +1
            #print ("score is: "+str(error_score))
        else:
            false_negative = false_negative +1
            #print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "5"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "3"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "4"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        (result, error_score, input_transform,something) = multiperson.match(model_features, input_features, True)
        if error_score < error_score_tresh:
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
    print ("error_score_tresh: "+str(error_score_tresh))

    return (precision,recall)

def make_pr_curve(pose):
    global eucl_dis_tresh_torso
    global rotation_tresh_torso
    global eucl_dis_tresh_legs
    global rotation_tresh_legs
    global eucld_dis_shoulders_tresh

    eucl_dis_tresh_torso= 0.098
    rotation_tresh_torso= 10.847
    eucl_dis_tresh_legs= 0.05
    rotation_tresh_legs= 14.527
    eucld_dis_shoulders_tresh= 0.085

    precisions = [];
    recalls = []
    start_error_tresh = 0
    for i in range(0,60):
        error_tresh = start_error_tresh + 0.02*i
        (precision,recall) = calculate_pr(pose,error_tresh)
        precisions.append(precision)
        recalls.append(recall)

    return (precisions,recalls)


def draw_pr_curve():
    (precission,recall) = make_pr_curve("1")
    (angle_precission,angle_recall) = make_pr_curve_angle("1")
    plt.plot(recall,precission)
    plt.plot(angle_recall, angle_precission)

    plt.legend(['Affine transformation', 'Angle script'], loc='center left')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('Pr curve of pose : 1')
    plt.axis([0,1,0,1])
    plt.legend()

    plt.show()

#**********************************pr curve angles*******************************
def make_pr_curve_angle(pose):

    precisions = [];
    recalls = []
    start_error_tresh = 0
    for i in range(0,100):
        '''
        eucl_dis_tresh_torso= start_eucl_dis_tresh_torso +(i*0.001)
        rotation_tresh_torso= start_rotation_tresh_torso +(i*0.2)
        eucl_dis_tresh_legs= start_eucl_dis_tresh_legs +(i*0.001)
        rotation_tresh_legs= start_rotation_tresh_legs +(i*0.1)
        eucld_dis_shoulders_tresh= start_eucld_dis_shoulders_tresh +(i*0.001)
        '''
        error_tresh = start_error_tresh + 0.6*i
        (precision,recall) = calculate_pr_angle(pose,error_tresh)
        precisions.append(precision)
        recalls.append(recall)

    return (precisions,recalls)

def calculate_pr_angle(pose,error_score_tresh):
    '''
    path = poses+pose
    model = path+"/json/"+pose+".json"
    '''
    poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/poses/pose'
    pose = "1"
    path = poses+pose
    model = path+"/json/0.json"
    model_features = common.parse_JSON_multi_person(model)

    true_positive =0
    false_positive =0
    true_negative =0
    false_negative =0

    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            true_positive = true_positive +1
            #print ("score is: "+str(error_score))
        else:
            false_negative = false_negative +1
            #print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "2"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "3"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "4"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "5"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "6"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
#///////////////////////check for pose5
    pose = "2"
    path = poses+pose
    model = path+"/json/0.json"
    model_features = common.parse_JSON_multi_person(model)
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            true_positive = true_positive +1
            #print ("score is: "+str(error_score))
        else:
            false_negative = false_negative +1
            #print ("error at: "+json)

    for json in glob.iglob(path+"/jsonfout/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "5"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1

    pose = "3"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
            false_positive = false_positive +1
            #print ("error at: "+json)
        else:
            true_negative = true_negative +1
    pose = "4"
    path = poses+pose
    for json in glob.iglob(path+"/json/*.json"):
        #print (json)
        input_features = common.parse_JSON_multi_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.succes(primary_angles, secondary_angles,error_score_tresh)
        if result:
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
    print ("error_score_tresh: "+str(error_score_tresh))

    return (precision,recall)
