import numpy as np
import numba as nb

numba_operator_wrapper = nb.njit(nb.float64[:](nb.float64[:, :]))
numba_operator_with_grad_wrapper = nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(nb.float64[:, :]))


numba_v2v_float_wrapper = nb.njit(nb.float64[:](nb.float64[:]))

numba_v2v_wrapper = nb.njit([nb.float64[:](nb.float64[:]),
                             nb.float64[:](nb.int64[:]),
                             nb.float64[:](nb.int32[:]),
                             nb.int64[:](nb.int64[:])])

@numba_v2v_wrapper
def normalize_norm1(x):
    
    return x / (np.sum(np.abs(x)) + 1e-12)

@numba_v2v_float_wrapper
def log_normalize(x):
    margin = np.abs(x)
    margin = np.where(margin > 1, 1 + np.log(margin), margin)
    sign = np.sign(x)
    return sign * margin


@nb.njit
def numba_randomchoice_w_prob( prob):
    assert np.abs(np.sum(prob) - 1.0) < 1e-9
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