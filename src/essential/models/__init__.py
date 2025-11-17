from .baselinear import BaseLinearModel
from .linearlowdim import LinearLowDimModel
from .linearlowdim2 import LinearLowDim2Model
from .linearmultiplicative import LinearMultiplicativeModel
from .linearhardko import LinearHardKoModel
from .linearhardkozeroorder import LinearHardKoZeroOrderModel
from .linearhardmultiplicative import LinearHardMultiplicativeModel
from .linearzeroorder import LinearZeroOrderModel
from .sigmoidhardkozeroorder import SigmoidHardKoZeroOrderModel
from .sigmoidhardko import SigmoidHardKoModel
from .sigmoid2 import Sigmoid2Model
from .linearmultiplicative import LinearMultiplicativeModel
from .lineardecay import LinearDecayModel
from .static import StaticModel
from .sigmoid2small import Sigmoid2SmallModel
from .sigmoid3 import Sigmoid3Model
from .tanh import TanhModel


MODEL_REGISTRY = {
    "linear": BaseLinearModel,
    "linearhardko": LinearHardKoModel,
    "linearhardkozeroorder": LinearHardKoZeroOrderModel,
    "linearzeroorder": LinearZeroOrderModel,
    "sigmoidhardkozeroorder": SigmoidHardKoZeroOrderModel,
    "sigmoidhardko": SigmoidHardKoModel,
    "sigmoid2": Sigmoid2Model,
    "sigmoid3": Sigmoid3Model,
    "linearhardmultiplicative": LinearHardMultiplicativeModel,
    "linearmultiplicative": LinearMultiplicativeModel,
    "lineardecay": LinearDecayModel,
    "linearlowdim": LinearLowDimModel,
    "linearlowdim2": LinearLowDim2Model,
    "static": StaticModel,
    "sigmoid2small": Sigmoid2SmallModel,
    "tanh": TanhModel,
}
