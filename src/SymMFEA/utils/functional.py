import numpy as np
from numba import jit

@jit(nopython = True)
def normalize(x):
    norm2 = np.sum(x ** 2, axis = 0)
    norm2 = np.sqrt(norm2) + 1e-12
    return x / norm2

@jit(nopython = True)
def numba_randomchoice_w_prob( prob):
    assert np.abs(np.sum(prob) - 1.0) < 1e-9
    rd = np.random.rand()
    res = 0
    sum_p = prob[0]
    while rd > sum_p:
        res += 1
        sum_p += prob[res]
    return res

@jit(nopython = True)
def numba_randomchoice(a, size= None, replace= True):
    return np.random.choice(a, size= size, replace= replace)

@jit(nopython = True)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 