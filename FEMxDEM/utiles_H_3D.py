import numpy as np
import math

def H_vars_1(param):
    '''
    calculate the history variables
    :param h_var:
    :return:
    '''
    return np.linalg.norm(np.array(param))

def H_vars_2(param):
    '''
    calculate the history variables
    :param h_var:
    :return:
    '''
    return np.mean(np.array(param))


def H_vars_3(param):
    '''
    calculate the history variables
    :param h_var:
    :return:
    '''
    J2 = (np.square(param[0] - param[3]) + np.square(param[0] - param[5]) + np.square(param[3] - param[5])) / 6 \
    + np.square(param[1]) + np.square(param[2]) + np.square(param[4])


    return math.sqrt(4 * J2[0] / 3)
    # return J2[0]

