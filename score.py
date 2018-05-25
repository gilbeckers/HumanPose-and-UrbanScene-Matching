import numpy as np

def get(x):
    the_score_function = -49.3518266 * np.log(x) - 65.30055317
    the_score = round(the_score_function, 1)

    if  the_score <= 55:
        the_score = the_score - the_score/7

    if the_score >= 100:
        print("ssssssore (in the_score)   %f", the_score)
        the_score = 99 - 4*(100/the_score)



    return  round(the_score, 1)