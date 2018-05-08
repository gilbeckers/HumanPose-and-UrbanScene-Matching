# source: https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
from os import walk
import common
import numpy as np
import json

path = "json_data/kever/"

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def rewrite_json(path, id):

    with open(path) as data_file:
        data = json.load(data_file)

    with open(path, 'w') as data_file:
        selected_pose = data["people"][id]
        print(selected_pose)

        #print(data)
        data["people"] = [selected_pose]
        #print(data)
        json.dump(data, data_file, indent=4)



    #print(data['people'])

    return


def filter_poses(poses, path):
    print("bigg: " , get_biggest(poses))
    rewrite_json(path,get_biggest(poses) )

    return

def get_biggest(poses):
    biggest = 0
    id = 0
    distance = 0  # afstand tussen neus en nek
    for pose in poses:
        A = pose[2]
        B = pose[5]

        if (A[0]==0 and A[1]==0) or (B[0]==0 and B[1]==0):
            #print("skip deze maffa")
            continue

        diff = np.abs(A - B)
        new_distance = dist(A, B)
        new_distance = ((diff[0]) ** 2 + diff[1] ** 2) ** 0.5

        if new_distance > distance:
            distance = new_distance
            biggest = id

        id = id+1

    return biggest



# test_file = path + "testje.json"
# poses = common.parse_JSON_multi_person(test_file)
# filter_poses(poses, test_file)



f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

print(f)
for f_name in f:
    poses = common.parse_JSON_multi_person(path + f_name)
    lenght = len(poses)

    if lenght>1:
        filter_poses(poses, path+f_name)

