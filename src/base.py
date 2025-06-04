# src/base.py - Updated with T-PVI parameters

from typing import (NamedTuple,
                    Callable,
                    Any)
import jax
from optax._src.base import (OptState,
                             GradientTransformation,
                             GradientTransformationExtraArgs,
                             EmptyState)
from src.id import PID, ID
from jax.random import PRNGKey

__all__ = ['PIDOpt',
           'SVIOpt',
           'SMOpt',
           'PIDCarry',
           'SVICarry',
           'SMCarry',
           'PIDParameters',
           'TPIDParameters',  # Added T-PVI parameters
           'TPIDCarry',       # Added T-PVI carry
           'SVIParameters',
           'Parameters',
           'Target',
           'ModelParameters',
           'ThetaOptParameters',
           'ROptParameters',
           'UVIParameters',
           'SMParameters',
           'DualParameters',]


class Target:
    de: bool
    def log_prob(self, x, y):
        raise NotImplementedError()


class RPreconParameters(NamedTuple):
    """
    Hyperparameters for the Preconditioner.
    """
    type: str
    max_norm: float
    agg: str


class ModelParameters(NamedTuple):
    d_z: int
    use_particles: bool
    n_hidden: int = 0
    d_y : int = 0
    kernel: str = 'fixed_diag_norm'
    n_particles: int=0


class ThetaOptParameters(NamedTuple):
    lr : float
    optimizer: str

    lr_decay: bool=False
    min_lr : float=0.
    interval : int=0.

    clip : bool=False
    max_clip : float=1

    regularization : float=0


class ROptParameters(NamedTuple):
    lr: float
    
    lr_decay: bool=False
    min_lr : float=0.
    interval : int=0.

    regularization: float=0

    n_samples: int=0


class DualParameters(NamedTuple):
    n_hidden: int


class Parameters(NamedTuple):
    algorithm: str
    model_parameters: ModelParameters
    theta_opt_parameters: ThetaOptParameters=None
    extra_alg_parameters: NamedTuple=None
    r_opt_parameters: ROptParameters=None
    r_precon_parameters: RPreconParameters=None
    dual_parameters: DualParameters=None
    dual_opt_parameters: ThetaOptParameters=None


class PIDOpt(NamedTuple):
    """
    A pair of State and Gradient Transformation.
    """
    theta_optim : GradientTransformation
    r_optim : GradientTransformation
    r_precon : Any = None


class SVIOpt(NamedTuple):
    theta_optim : GradientTransformation


class SMOpt(NamedTuple):
    theta_optim : GradientTransformation
    dual_optim : GradientTransformation


class PIDCarry(NamedTuple):
    id: PID
    theta_opt_state: OptState
    r_opt_state: OptState
    r_precon_state: OptState


class TPIDCarry(NamedTuple):
    """Extended PID Carry with temperature annealing state"""
    id: PID
    theta_opt_state: OptState
    r_opt_state: OptState
    r_precon_state: OptState
    # Temperature annealing state
    iteration: int = 0
    current_lambda_r: float = 1e-2
    annealing_stopped: bool = False
    entropy_history: jax.Array = None
    diversity_history: jax.Array = None


class CVState(NamedTuple):
    grads: jax.numpy.ndarray
    total: jax.numpy.ndarray


class SVICarry(NamedTuple):
    id: ID
    theta_opt_state: OptState


class SMCarry(NamedTuple):
    id: ID
    theta_opt_state: OptState
    dual: Callable
    dual_opt_state: OptState


class PIDParameters(NamedTuple):
    """
    Hyperparameters for PID.
    """
    fudge: float=0 # Fudge Factor. Denoted by Gamma in the paper
    mc_n_samples: int=250 # Number of Monte Carlo Samples for the Gradient Estimation


class TPIDParameters(NamedTuple):
    """Extended PID Parameters with temperature annealing"""
    fudge: float = 0
    mc_n_samples: int = 250
    # Temperature annealing parameters
    annealing_schedule: str = "polynomial"  # "polynomial", "exponential", "inverse_sigmoid"
    lambda_0: float = 1e-2  # Initial regularization strength
    lambda_min: float = 1e-8  # Minimum regularization (floor)
    gamma: float = 0.55  # Polynomial decay rate
    alpha: float = 0.01  # Exponential decay rate
    t0: float = 1000  # Sigmoid transition point
    tau: float = 100  # Sigmoid transition smoothness
    # Monitoring parameters
    monitor_entropy: bool = True
    entropy_threshold: float = 0.1  # Stop annealing if entropy drops too fast
    diversity_threshold: float = 0.01  # Stop annealing if particle diversity too low


class SVIParameters(NamedTuple):
    """
    Hyperparameters for SVI.
    """
    mc_n_samples: int = 250 # Number of Monte Carlo Samples for the Gradient Estimation
    K: int = 50 # TODO


class UVIParameters(NamedTuple):
    """
    Hyperparameters for UVI.
    """
    mc_n_samples: int = 250 # Number of Monte Carlo Samples for the Gradient Estimation


class SMParameters(NamedTuple):
    """
    Hyperparameters for SM.
    """
    dual_steps: int = 1 # Number of Dual Steps
    train_steps: int = 1