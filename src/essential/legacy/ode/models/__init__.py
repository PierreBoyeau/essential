from .sigmoid2_ode import Sigmoid2ODEModel
from .sigmoid2_steady_state import Sigmoid2SteadyStateModel
from .hardsigmoid2_steady_state import HardSigmoid2SteadyStateModel
from .hardsigmoid2_embedding_steady_state import HardSigmoid2EmbeddingSteadyStateModel
from .sigmoid2_flow import Sigmoid2FlowModel
from .static import StaticModel


MODEL_REGISTRY = {
    "sigmoid2_ode": Sigmoid2ODEModel,
    "sigmoid2_steady_state": Sigmoid2SteadyStateModel,
    "hardsigmoid2_steady_state": HardSigmoid2SteadyStateModel,
    "hardsigmoid2_embedding_steady_state": HardSigmoid2EmbeddingSteadyStateModel,
    "sigmoid2_flow": Sigmoid2FlowModel,
    "static": StaticModel,
}
