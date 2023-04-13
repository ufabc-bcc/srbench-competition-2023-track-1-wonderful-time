from .sum import Sum
from .subtract import Subtract
from .operand import Operand
from .node import Node
from .tanh import Tanh
from .prod import Prod
from .relu import Relu
from .aq import AQ
from .constant import Constant
from .log import Log
from .valve import Valve
from .batchnorm import BatchNorm
FUNCTION_SET = {
    0 : [
        Operand,
        Constant,
    ],
    1: [
        Tanh,
        Log,
        Relu,
        BatchNorm,
    ],
    2: [
        Prod,
        AQ,
        Valve,
    ],
    -1:[
        Sum,
        Subtract, 
    ]
}

LINEAR_FUNCTION_SET = {k: [f for f in v if not f.is_nonlinear]  for k, v in FUNCTION_SET.items() if len([f for f in v if not f.is_nonlinear])}


