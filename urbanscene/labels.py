posegroups= {
    "dart":[
        [1,2,3,4,6,7],
        [1, 2,13, 4, 7],
        [8, 10, 11, 12],
        [1, 9,2],
        [5],
        [9]
    ]
}


def chech_same_class(group, x,y):
    if group in posegroups:
        classes = posegroups[group]
        for a in classes:
            if x in a and y in a:
                return  True
            #elif (x in a and y not in a) or (x not in a and y in a):
            #    return False

        return False
    else:
        return True