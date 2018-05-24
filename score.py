import numpy as np
max_euclidean_error_norm = 0.08
x = round(max_euclidean_error_norm,4)*100
print(x)
a = 2
b = 0
c = -26.75243254
d = 3.2
x=0.08

# y = -49.3518266 ln(x) - 65.30055317
the_score_function = -49.3518266*np.log(x) - 65.30055317
the_score_function = 7.000825 + (1.099995 - 7.000825)/(1 + (x/3.072862)**31.03071)
#the_score_function = c*np.log( (a*x) / d)

print(the_score_function)
