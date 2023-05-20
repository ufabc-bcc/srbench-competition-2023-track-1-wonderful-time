from SymMFEA.components.functions import *
from SymMFEA.utils.functional import *
from SymMFEA.components.trainer.loss import *
import numpy as np
import pytest
from .utils import *
import random


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
        
        
