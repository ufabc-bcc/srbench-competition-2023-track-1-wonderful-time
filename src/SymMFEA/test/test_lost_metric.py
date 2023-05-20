from SymMFEA.components.functions import *
from SymMFEA.utils.functional import *
from SymMFEA.components.trainer.loss import *
from SymMFEA.components.metrics import *
import numpy as np
import pytest
from .utils import *
from sklearn.metrics import log_loss

class TestLost():
    
    @pytest.mark.parametrize("x, y", [
        (3, 0.9525741268224809),
        (-18.5, 9.237449576580103e-9),
        (0.76, 0.6813537337890835),
        (-1.9, 0.13010847436292233),
    ])
    def test_sigmoid(self, x:float, y:float):
        y_hat = sigmoid(np.array([x], dtype=np.float64)).item()
        assert abs(y - y_hat) < 1e-5
        
        
class TestMetric():
    l = LogLoss()
    @pytest.mark.parametrize("Y", generate_input_list((2, 10), 10))
    def test_log_loss(self, Y:np.ndarray):
        
        y_hat, y = Y[0], Y[1]
        
        y = sigmoid(y)
        y = np.where(y > 0.5, 1, 0)
        print(y_hat, y)
        
        l1 = log_loss(y, sigmoid(y_hat))
        l2 = logloss(y, y_hat)
        
        assert np.allclose(l1, l2)
    
    
    @pytest.mark.parametrize("x1, x2, gt", [
        (0.8, 0.9525741268224809, True),
        (2, 9.237449576580103e-9, False),
        (0.4, 0.6813537337890835, True),
        (0.1, 0.13010847436292233, True),
        (1, 0.5, False),
    ])
    def test_log_loss_is_better(self, x1, x2, gt):
        
        is_better = self.l.is_better(x1, x2)
        assert is_better == gt