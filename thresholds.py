#lel

OPENPOSE_ZEKERHEID = 0.2
OPENPOSE_AMOUNT_KEYPOINTS =2

'''------- SINGLE POSE PARAMETERS -----------'''
SP_DISTANCE_TORSO = 0.236#0.18 # 0.098
SP_ROTATION_TORSO = 19.8

SP_DISTANCE_LEGS = 0.082#0.058
SP_ROTATION_LEGS = 24   # 14.527

SP_DISTANCE_SHOULDER = 0.125



'''------- MULTIPLE POSE PARAMETERS -----------'''
MP_DISCTANCE = 0.38#0.13

'''-------- FEATURE MATCHING ------------------'''
MIN_MATCH_COUNT     = 16
FLANN_INDEX_KDTREE  = 1
FLANN_INDEX_LSH     = 6
FILTER_RATIO        = 0.8 #lagere ratio geeft minder 'good' matches


'''------- URBANSCENE MATCHING ---------------'''
PERSPECTIVE_CORRECTION = True
USI_AMOUNT_ITERATIONS = 1
AFFINE_TRANS_WHOLE_DISTANCE = 0.084
AMOUNT_BACKGROUND_FEATURES = 5  # note: moet kleiner zijn dan MIN_MATCH_COUNT!!
CROP = False
