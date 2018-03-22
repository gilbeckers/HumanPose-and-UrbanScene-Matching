#lel

'''------- SINGLE POSE PARAMETERS -----------'''
SP_DISTANCE_TORSO = 0.18 # 0.098
SP_ROTATION_TORSO = 19

SP_DISTANCE_LEGS = 0.058
SP_ROTATION_LEGS = 24   # 14.527

SP_DISTANCE_SHOULDER = 0.125



'''------- MULTIPLE POSE PARAMETERS -----------'''
MP_DISCTANCE = 0.13

'''-------- FEATURE MATCHING ------------------'''
MIN_MATCH_COUNT     = 10
FLANN_INDEX_KDTREE  = 1
FLANN_INDEX_LSH     = 6
FILTER_RATIO        = 0.8 #lagere ratio geeft minder 'good' matches


'''------- URBANSCENE MATCHING ---------------'''
AFFINE_TRANS_WHOLE_DISTANCE = 0.1
AMOUNT_BACKGROUND_FEATURES = 4   # note: moet kleiner zijn dan MIN_MATCH_COUNT!!
