import numpy as np
import numba as nb
import os
EPS = np.finfo(np.float32).eps
ONE = np.float32(1)
ZERO = np.float32(0)
numba_operator_wrapper = nb.njit(nb.float32[:](nb.float32[:, :]), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None, nogil=True)
numba_operator_with_grad_wrapper = nb.njit(nb.types.Tuple((nb.float32[:], nb.float32[:, :]))(nb.float32[:, :]), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)


numba_v2v_float_wrapper = nb.njit([nb.float32[:](nb.float32[:]),
                                   nb.float64[:](nb.float64[:])], cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)

numba_v2v_int_wrapper = nb.njit(nb.int64[:](nb.float32[:]), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)


numba_v2v_wrapper = nb.njit([nb.float32[:](nb.float32[:]),
                             nb.float32[:](nb.int64[:]),
                             nb.float32[:](nb.int32[:]),
                             nb.float64[:](nb.float64[:]),
                             nb.int64[:](nb.int64[:])], cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)

numba_update_adam = nb.njit(nb.types.Tuple((nb.float32[:], nb.float32[:]))(nb.float32[:], nb.float32, nb.float32[:], nb.int64),
    cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)

nb.njit(nb.float32[:](nb.float32[:], nb.float32), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def multiply(vec, val):
    return vec * val


@nb.njit(nb.int64(nb.float32[:]))
def nb_argmax(x):
    return np.argmax(x)



@numba_v2v_wrapper
def normalize_norm1(x):
    
    return x / np.float32((np.sum(np.abs(x)) + EPS))

@numba_v2v_float_wrapper
def log_normalize(x):
    margin = np.abs(x)
    margin = np.where(margin > ONE, ONE + np.log(margin), margin)
    sign = np.sign(x)
    return sign * margin


@nb.njit
def numba_randomchoice_w_prob(prob):
    assert np.abs(np.sum(prob) - 1.0) < 1e-2
    rd = np.random.rand()
    res = 0
    sum_p = prob[0]
    while rd > sum_p:
        res += 1
        sum_p += prob[res]
    return res

@nb.njit
def numba_randomchoice(a, size, replace= True):
    return np.random.choice(a, size= size, replace= replace)

@numba_v2v_float_wrapper
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

@numba_v2v_float_wrapper
def sigmoid(x):
    x = np.clip(x, -20, 20)
    return np.where(x >= ZERO, ONE / (ONE + np.exp(-x)), np.exp(x) / (ONE + np.exp(x))).astype(x.dtype)