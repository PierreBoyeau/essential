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


MODEL_REGISTRY = {
    "linear": BaseLinearModel,
    "linearhardko": LinearHardKoModel,
    "linearhardkozeroorder": LinearHardKoZeroOrderModel,
    "linearzeroorder": LinearZeroOrderModel,
    "sigmoidhardkozeroorder": SigmoidHardKoZeroOrderModel,
    "sigmoidhardko": SigmoidHardKoModel,
    "sigmoid2": Sigmoid2Model,
    "linearhardmultiplicative": LinearHardMultiplicativeModel,
    "linearmultiplicative": LinearMultiplicativeModel,
    "lineardecay": LinearDecayModel,
    "linearlowdim": LinearLowDimModel,
    "linearlowdim2": LinearLowDim2Model,
    "static": StaticModel,
}
