from .steady_state_forcing import SteadyStateForcingModel
from .steady_state_decay import SteadyStateDecayModel
from .dynamic_cellbox import DynamicCellboxModel
from .dynamic_cellboxlowdim import DynamicCellboxLowDimModel
from .dynamic_cellboxlowdim2 import DynamicCellboxLowDimModel2
from .dynamic_hardmultiplicative import DynamicHardMultiplicativeModel
from .dynamic_hardko import DynamicHardKoModel
from .dynamic_hardkozeroorder import DynamicHardKoZeroOrderModel
from .dynamic_cellboxzeroorder import DynamicCellboxZeroOrderModel
from .dynamic_sigmoidhardkozeroorder import DynamicSigmoidHardKoZeroOrderModel
from .dynamic_multiplicative import DynamicMultiplicativeModel
from .dynamic_decay import DynamicDecayModel
from .static import StaticModel


MODEL_REGISTRY = {
    "steady_state_forcing": SteadyStateForcingModel,
    "steady_state_decay": SteadyStateDecayModel,
    "dynamic_cellbox": DynamicCellboxModel,
    "dynamic_hardko": DynamicHardKoModel,
    "dynamic_hardkozeroorder": DynamicHardKoZeroOrderModel,
    "dynamic_cellboxzeroorder": DynamicCellboxZeroOrderModel,
    "dynamic_sigmoidhardkozeroorder": DynamicSigmoidHardKoZeroOrderModel,
    "dynamic_cellboxlowdim": DynamicCellboxLowDimModel,
    "dynamic_cellboxlowdim2": DynamicCellboxLowDimModel2,
    "dynamic_hardmultiplicative": DynamicHardMultiplicativeModel,
    "dynamic_multiplicative": DynamicMultiplicativeModel,
    "dynamic_decay": DynamicDecayModel,
    "static": StaticModel,
}
