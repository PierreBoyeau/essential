from .steady_state_forcing import SteadyStateForcingModel
from .steady_state_decay import SteadyStateDecayModel
from .multiplicative_knockdown import MultiplicativeKnockdownModel
from .multiplicative_knockdown_with_basal import MultiplicativeKnockdownWithBasal
from .dynamic_cellbox import DynamicCellboxModel
from .dynamic_hardmultiplicative import DynamicHardMultiplicativeModel
from .dynamic_multiplicative import DynamicMultiplicativeModel
from .dynamic_linear import DynamicLinearModel
from .dynamic_linear_softplus import DynamicLinearSoftplusModel


MODEL_REGISTRY = {
    "steady_state_forcing": SteadyStateForcingModel,
    "steady_state_decay": SteadyStateDecayModel,
    "multiplicative_knockdown": MultiplicativeKnockdownModel,
    "multiplicative_knockdown_with_basal": MultiplicativeKnockdownWithBasal,
    "dynamic_cellbox": DynamicCellboxModel,
    "dynamic_hardmultiplicative": DynamicHardMultiplicativeModel,
    "dynamic_multiplicative": DynamicMultiplicativeModel,
}
