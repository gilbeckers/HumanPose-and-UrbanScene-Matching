import numpy as np

def scene_feature_scaling(input, xmax, ymax):

    xmin = 0
    ymin = 0

    sec_x = (input[:, 0] - xmin) / (xmax - xmin)
    sec_y = (input[:, 1] - ymin) / (ymax - ymin)

    output = np.vstack([sec_x, sec_y]).T
    #output[output < 0] = 0
    #logger.info("out: %s", str(output))
    return output

#Cut pose out of image
def feature_scaling(input):
    #logger.info("inn: %s" , str(input))
    # We accept the presence of (0,0) points in the input poses (undetected body-parts)
    # But we don't want them to influence our normalisation

    # Here it's assumed that (0,y) and (x,0) don't occur
    # Is a acceptable assumption because the chance is sooooo small
    #   that a feature is positioned just right on the x or y axis
    xmax = max(input[:, 0])
    ymax = max(input[:, 1])

    xmin = np.min(input[np.nonzero(input[:,0])]) #np.nanmin(input[:, 0])
    ymin = np.min(input[np.nonzero(input[:,1])]) #np.nanmin(input[:, 1])

    sec_x = (input[:, 0]-xmin)/(xmax-xmin)
    sec_y = (input[:, 1]-ymin)/(ymax-ymin)

    output = np.vstack([sec_x, sec_y]).T
    output[output<0] = 0
    #logger.info("out: %s", str(output))
    return output

def feature_scaling_multi_person(input):
    normalized = []
    xmax = 0
    ymax = 0
    i = 0
    while np.min(input[i][np.nonzero(input[i][:,0])][:, 0]) ==0:
        i = i+1
    xmin =np.min(input[i][np.nonzero(input[i][:,0])][:, 0])
    i =0
    while np.min(input[i][np.nonzero(input[i][:,1])][:, 1]) ==0:
        i= i+1
    ymin = np.min(input[i][np.nonzero(input[i][:,1])][:, 1])
    for pose in input:

        xmax_pose = max(pose[:, 0])
        ymax_pose = max(pose[:, 1])
        xmin_pose = np.min(pose[np.nonzero(pose[:,0])][:, 0])
        ymin_pose = np.min(pose[np.nonzero(pose[:,1])][:, 1])
        #print (pose[np.nonzero(pose[:,1])][:, 0])

        if xmax_pose > xmax:
            xmax =xmax_pose
        if ymax_pose > ymax:
            ymax =ymax_pose
        if xmin_pose < xmin:
            xmin =xmin_pose
        if ymin_pose < ymin:
            ymin =ymin_pose

    for pose in input:
        sec_x = (pose[:, 0]-xmin)/(xmax-xmin)
        sec_y = (pose[:, 1]-ymin)/(ymax-ymin)

        normalized.append(np.vstack([sec_x, sec_y]).T)

    normalized = np.array(normalized)
    #print("xmax: "+str(xmax)+" ymax: "+str(ymax)+" xmin: "+str(xmin)+" ymin: "+str(ymin))
    return normalized


def divide_by_max(input):
    xmax = max(input[:, 0])
    ymax = max(input[:, 1])

    xmin = min(input[:, 0])
    ymin = min(input[:, 1])

    #sec_x = (input[:, 0]-xmin)/(xmax-xmin)
    #sec_y = (input[:, 1]-ymin)/(ymax-ymin)

    sec_x = (input[:, 0]) / (xmax)
    sec_y = (input[:, 1]) / (ymax)

    output = np.vstack([sec_x, sec_y]).T

    return output




def normalise_rescaling(input):
    xmax = max(input[:, 0])
    xmin = min(input[:, 0])
    ymax = max(input[:, 1])
    ymin = min(input[:, 1])

    sec_x = (input[:, 0] - xmin) / (xmax - xmin)
    sec_y = (input[:, 1] - ymin) / (ymax - ymin)
    output = np.vstack([sec_x, sec_y]).T

    return output

def normalise_standardization(input):
    xmean = input[:,0].mean(axis=0)
    ymean = input[:,1].mean(axis=0)
    xstd = np.std(input[:,0])
    ystd = np.std(input[:, 1])

    sec_x = (input[:, 0] - xmean) / xstd
    sec_y = (input[:, 1] - ymean) / ystd
    output = np.vstack([sec_x, sec_y]).T

    return output
