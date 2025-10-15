from .steady_state_forcing import SteadyStateForcingModel
from .steady_state_decay import SteadyStateDecayModel
from .multiplicative_knockdown import MultiplicativeKnockdownModel
from .multiplicative_knockdown_with_basal import MultiplicativeKnockdownWithBasal
from .dynamic_cellbox import DynamicCellboxModel

MODEL_REGISTRY = {
    "steady_state_forcing": SteadyStateForcingModel,
    "steady_state_decay": SteadyStateDecayModel,
    "multiplicative_knockdown": MultiplicativeKnockdownModel,
    "multiplicative_knockdown_with_basal": MultiplicativeKnockdownWithBasal,
    "dynamic_cellbox": DynamicCellboxModel,
}
