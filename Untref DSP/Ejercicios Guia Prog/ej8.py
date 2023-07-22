import numpy as np

def mean(x):
    return sum(x)/len(x)

def des_est(x):
    u = mean(x)
    sum_terms = 0
    for i in range(len(x)):
        sum_terms += (x[i] - u)**2

    sum_terms /= len(x) - 1

    return np.sqrt(sum_terms)

def rms_value(x):
    sum_terms = 0
    for i in range(len(x)):
        sum_terms += (x[i])**2

    sum_terms /= len(x)

    return np.sqrt(sum_terms)
    