#!/usr/bin/env python

'''
This module contains some common routines used by other samples.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2 as cv
import json
import logging
logger = logging.getLogger("common")

# built-in modules
import os
import itertools as it
from contextlib import contextmanager

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

class Bunch(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __str__(self):
        return str(self.__dict__)

import numpy as np

def find_transformation(model_features, input_features):
    # Zoek 2D affine transformatie matrix om scaling, rotatatie en translatie te beschrijven tussen model en input
    # 2x2 matrix werkt niet voor translaties

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]


    # It needs to be checked if a (0,0) row is present due to undetected body-parts
    # Initially undetected features are accepted in both the input & model pose
    # But, before finding the affine transformation these are filterd out
    input_counter = 0

    # List with indices of all the (0,0)-rows
    # This is important because they need to
    # be removed before finding the affine transformation
    # But before returning to caller, they should be restored at the same place.
    # Because the correspondence of the points needs to be preserved
    nan_indices = []

    #print("inputttt: " , input_features)

    input_features_zonder_nan = []
    model_features_zonder_nan = []
    for in_feature in input_features:
        if (in_feature[0] == 0) and (in_feature[1] == 0): # is a (0,0) feature
            nan_indices.append(input_counter)
        else:
            input_features_zonder_nan.append([in_feature[0], in_feature[1]])
            model_features_zonder_nan.append([model_features[input_counter][0], model_features[input_counter][1]])
        input_counter = input_counter+1

    if len(model_features_zonder_nan)==0 or len(input_features_zonder_nan) ==0:
        #print("################## hiereeeeee")
        return (input_features,[])

    input_features = np.array(input_features_zonder_nan)
    model_features = np.array(model_features_zonder_nan)

    # padden:
    # naar vorm [ x x 0 1]
    Y = pad(model_features)
    X = pad(input_features)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A and then we can display the input on the model = Y'
    A, res, rank, s = np.linalg.lstsq(X, Y)
    transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = transform(input_features)

    # Restore the (0,0) rows
    # TODO: maybe too much looping ..
    # TODO: convert van matrix->list->matrix ?? crappy
    # Note!: werkt enkel goed als nan_indices gesort is van klein naar groot!! anders kans over index out-of-bounds
    input_transform_list  = input_transform.tolist()
    for index in nan_indices:
        input_transform_list.insert(index, [0,0])
    input_transform = np.array(input_transform_list)


    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return (input_transform, A)



#TODO oude functie voor case waar enkel transformatie voor de fixed-points wordt berekend. (OUD)
def calcTransformationMatrix_fixed_points(model, input, secondary):

    # Zoek 2D affine transformatie matrix om scaling, rotatatie en translatie te beschrijven tussen model en input
    # 2x2 matrix werkt niet voor translaties
    # Pad the data with ones, so that our transformation can do translations too
    n = model.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
    unpad = lambda x: x[:, :-1]

    # padden:
    # naar vorm [ x x 0 1]
    X = pad(model)
    Y = pad(input)

    # print(X)
    # print(Y)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))
    #modelTransform = transform(model)
    modelTransform = transform(secondary)

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return (modelTransform, A)

'''
Description parse_JSON_single_person(filename)
Parse the openpose json output and returns an numpy array of 18 rows (body -joint points / keypoints)
so undetected body parts (openpose errors, labeled by openpose as (0,0) )
-> stay (0,0) and can be identified in this way

Parameters:
@:param filename

Returns:
@:returns a numpy array containg 18 2D features
'''
def parse_JSON_single_person(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    #Keypoints
    keypointsPeople1 = data["people"][0]["pose_keypoints"] #enkel 1 persoon  => [0]

    #18 2D coordinatenkoppels (joint-points)
    array = np.zeros((18,2))
    #list = []
    arrayIndex = 0

    for i in range(0, len(keypointsPeople1), 3):
        array[arrayIndex][0] = keypointsPeople1[i]
        array[arrayIndex][1] = keypointsPeople1[i+1]
        arrayIndex+=1

        #feature = [keypointsPeople1[i], keypointsPeople1[i+1]]
        #list.append(feature)

    return array
    #return list

# the json file is a string var that is currently loaded in memory
# The json file isn't read in this case
def parse_JSON_single_person_as_json(filename):
    #data = json.load(filename)
    data = filename
    #Keypoints
    keypointsPeople1 = data["people"][0]["pose_keypoints"] #enkel 1 persoon  => [0]

    #18 2D coordinatenkoppels (joint-points)
    array = np.zeros((18,2))
    #list = []
    arrayIndex = 0

    for i in range(0, len(keypointsPeople1), 3):
        array[arrayIndex][0] = keypointsPeople1[i]
        array[arrayIndex][1] = keypointsPeople1[i+1]
        arrayIndex+=1

        #feature = [keypointsPeople1[i], keypointsPeople1[i+1]]
        #list.append(feature)

    return array
    #return list

def parse_JSON_multi_person_old(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    list_of_features = []

    keypoints = data["people"]
    for k in range(0, len(keypoints)):
        person_keypoints = keypoints[k]["pose_keypoints"]

        # 18 3D coordinatenkoppels (joint-points)
        array = np.zeros((18, 2))
        arrayIndex = 0
        for i in range(0, len(person_keypoints), 3):
            array[arrayIndex][0] = person_keypoints[i]
            array[arrayIndex][1] = person_keypoints[i + 1]
            arrayIndex += 1
        list_of_features.append(array)

    return list_of_features

def parse_JSON_multi_person(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    list_of_features = []

    keypoints = data["people"]
    for k in range(0, len(keypoints)):
        person_keypoints = keypoints[k]["pose_keypoints"]

        # 18 3D coordinatenkoppels (joint-points)
        array = np.zeros((18, 2))
        arrayIndex = 0
        for i in range(0, len(person_keypoints), 3):
            if person_keypoints[i+2]> 0.18:  # was 0.25 was 0.4
                array[arrayIndex][0] = person_keypoints[i]
                array[arrayIndex][1] = person_keypoints[i+1]
            else:
                logger.warning("openpose certainty(%f) to low index: %d", person_keypoints[i+2], arrayIndex )
                array[arrayIndex][0] = 0
                array[arrayIndex][1] = 0
            arrayIndex+=1
        list_of_features.append(array)

    return list_of_features

def parse_JSON_multi_person_as_json(filename):
    data = filename

    list_of_features = []

    keypoints = data["people"]
    for k in range(0, len(keypoints)):
        person_keypoints = keypoints[k]["pose_keypoints"]

        # 18 3D coordinatenkoppels (joint-points)
        array = np.zeros((18, 2))
        arrayIndex = 0
        for i in range(0, len(person_keypoints), 3):
            array[arrayIndex][0] = person_keypoints[i]
            array[arrayIndex][1] = person_keypoints[i + 1]
            arrayIndex += 1
        list_of_features.append(array)

    return list_of_features

def split_in_face_legs_torso(features):
    # torso = features[2:8]   #zonder nek
    torso = features[1:8]   #met nek  => if nek incl => compare_incl_schouders aanpassen!!
    legs = features[8:14]
    face = np.vstack([features[0], features[14:18]])

    return (face, torso, legs)

def split_in_face_legs_torso_v2(features):
    # torso = features[2:8]   #zonder nek
    torso = features[[0,1,2,3,4,5,6,7,8,11]]   #met nek  => if nek incl => compare_incl_schouders aanpassen!!
    legs = features[[1,8,9,10,11,12,13]]
    face = np.vstack([features[0], features[14:18]])
    return (face, torso, legs)

def unsplit(face, torso, legs):
    whole = np.vstack([face[0], torso, legs, face[1:5]])

    return whole

def handle_undetected_points(input_features, model_features):
    # Because np.array is a mutable type => passed by reference
    #   -> dus als model wordt veranderd wordt er met gewijzigde array
    #       verder gewerkt na callen van single_person()
    # model_features_copy = np.array(model_features)
    model_features_copy = model_features.copy()
    input_features_copy = input_features.copy()


    # Input is allowed to have a certain amount of undetected body parts
    # In that case, the corresponding point from the model is also changed to (0,0)
    #   -> afterwards matching can still proceed
    # The (0,0) points can't just be deleted because
    # without them the feature-arrays would become ambigu. (the correspondence between model and input)
    #
    # !! NOTE !! : the acceptation and introduction of (0,0) points
    # is a danger for our current normalisation
    # These particular origin points should not influence the normalisation
    # (which they do if we neglect them, xmin and ymin you know ... )
    if np.any(input_features[:] == [0, 0]):
        counter = 0
        for feature in input_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                logger.debug(" Undetected body part in input: index(%d) %s", counter,
                               get_bodypart(counter))
                model_features_copy[counter][0] = 0
                model_features_copy[counter][1] = 0
                # input_features[counter][0] = 0#np.nan
                # input_features[counter][1] = 0#np.nan
            counter = counter + 1

    # In this second version, the model is allowed to have undetected features
    if np.any(model_features[:] == [0, 0]):
        counter = 0
        for feature in model_features:
            if feature[0] == 0 and feature[1] == 0:  # (0,0)
                logging.debug(" Undetected body part in MODEL: index(%d) %s", counter,
                               get_bodypart(counter))
                input_features_copy[counter][0] = 0
                input_features_copy[counter][1] = 0
            counter = counter + 1

    assert len(model_features_copy) == len(input_features_copy)

    # Normalise features: crop => delen door Xmax & Ymax (NIEUWE MANIER!!)
    # !Note!: as state above, care should be taken when dealing
    #   with (0,0) points during normalisation
    #
    # TODO:
    # !Note2!: The exclusion of a feature in the torso-regio doesn't effect
    #   the affine transformation in the legs- and face-regio in general.
    #   BUT in some case it CAN influence the (max-)euclidean distance.
    #     -> (so could resolve in different MATCH result)
    #   This is the case when the undetected bodypart [=(0,0)] would be the
    #   minX or minY in the detected case.
    #   Now, in the absence of this minX or minY, another feature will deliver
    #   this value.
    #   -> The normalisation region is smaller and gives different values after normalisation.
    #
    #   (BV: als iemand met handen in zij staat maar de rechter ellenboog niet gedetect wordt
    #       => minX is nu van het rechthand dat in de zij staat.

    # TODO
    # It seems like the number of excluded features is proportional with the rotation angle
    # -> That is, the more features are missing, the higher the rotation angle becomes, this is weird
    # -> NIET ECHT RAAR EIGENLIJK WANT MINDER punten betekent minder constraints, waardoor er meer kan gedraaid worden (meer vrijheidsgraad)

    return (input_features_copy, model_features_copy)


def unpad(matrix):

    return matrix[:, :-1]

def pad(matrix):
    return np.hstack([matrix, np.zeros((matrix.shape[0], 1))])

import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

options = {0 : 'neus',
           1 : 'nek',
           2 : 'l-schouder',
           3 : 'l-elleboog',
           4 : 'l-pols',
           5 : 'r-schouder',
           6 : 'r-elleboog',
           7 : 'r-pols',
           8 : 'l-heup',
           9 : 'l-knie',
           10: 'l-enkel',
           11: 'r-heup',
           12: 'r-knie',
           13: 'r-enkel',
           14: 'l-oog',
           15: 'r-oog',
           16: 'l-oor',
           17: 'r-oor',
        }

def get_bodypart(index):

    if(index <=17 and index >=0):
        return options[index]

    return 'no-bodypart (wrong index)'


def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv.INTER_AREA
    else: # stretching image
        interp = cv.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set paposematching.d color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)

    return scaled_img

def resize_img(model_image, input_image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 500.0 / model_image.shape[1]
    dim = (500, int(model_image.shape[0] * r))

    # perform the actual resizing of the image and show it
    model_image = cv.resize(model_image, dim, interpolation = cv.INTER_AREA)
    input_image = cv.resize(input_image, dim, interpolation = cv.INTER_AREA)

    return (model_image, input_image)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    returnposematching. path, name, ext

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def homotrans(H, x, y):
    xs = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    ys = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    s  = H[2, 0]*x + H[2, 1]*y + H[2, 2]
    return xs/s, ys/s

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx,  0, tx],
                    [  0, cy, ty],
                    [  0,  0,  1]])
    return M


def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)

def nothing(*arg, **kw):
    pass

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    if PY3:
        img0 = next(imgs)
    else:
        img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv.circle(vis, (int(x), int(y)), 2, color)


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

def feature_scaling_multi_person():

    return


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
