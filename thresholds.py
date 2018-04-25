#lel
'''------- COMMON PARAMETER ---------'''
CP_ACCURACY = 0.1  # was 0.18 <-- 0.25 <-- 0.4


'''------- SINGLE POSE PARAMETERS -----------'''
SP_DISTANCE_FACE = 1
SP_DISTANCE_TORSO = 0.30 # 0.098
SP_ROTATION_TORSO = 30

SP_DISTANCE_LEGS = 1.3
SP_ROTATION_LEGS = 24   # 14.527

SP_DISTANCE_SHOULDER = 0.2



'''------- MULTIPLE POSE PARAMETERS -----------'''
MP_DISCTANCE = 0.5
MP_ERROR_DISTANCE = 2.6

'''-------- FEATURE MATCHING ------------------'''
MIN_MATCH_COUNT     = 10
FLANN_INDEX_KDTREE  = 1
FLANN_INDEX_LSH     = 6
FILTER_RATIO        = 0.8 #lagere ratio geeft minder 'good' matches


'''------- URBANSCENE MATCHING ---------------'''
AFFINE_TRANS_WHOLE_DISTANCE = 0.084
AMOUNT_BACKGROUND_FEATURES = 9   # note: moet kleiner zijn dan MIN_MATCH_COUNT!!
