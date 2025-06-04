# src/trainers/t_pvi.py - Fixed for empty array indexing issues

import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple, Callable
from src.base import (Target,
                      TPIDCarry,
                      PIDOpt,
                      TPIDParameters)
from jaxtyping import PyTree


def polynomial_annealing(lambda_0: float, t: int, gamma: float = 0.55):
    """Polynomial decay: lambda_r(t) = lambda_0 * (1 + t)^(-gamma)"""
    return lambda_0 * (1 + t) ** (-gamma)


def exponential_annealing(lambda_0: float, t: int, alpha: float = 0.01):
    """Exponential decay: lambda_r(t) = lambda_0 * exp(-alpha * t)"""
    return lambda_0 * np.exp(-alpha * t)


def inverse_sigmoid_annealing(lambda_0: float, t: int, t0: float = 1000, tau: float = 100):
    """Inverse sigmoid: lambda_r(t) = lambda_0 / (1 + exp((t - t0)/tau))"""
    return lambda_0 / (1 + np.exp((t - t0) / tau))


def compute_lambda_r(hyperparams: TPIDParameters, t: int):
    """Compute regularization strength - simplified for JAX compatibility"""
    # Use polynomial annealing as default for JAX compatibility
    lambda_r = polynomial_annealing(hyperparams.lambda_0, t, hyperparams.gamma)
    return np.maximum(lambda_r, hyperparams.lambda_min)


def compute_particle_diagnostics(particles: jax.Array):
    """Compute particle diagnostics - safe version"""
    # Ensure we have valid particles
    n_particles, dim = particles.shape
    
    # Simple entropy estimate based on particle spread
    particle_std = np.std(particles, axis=0)
    entropy = np.sum(np.log(particle_std + 1e-8))
    
    # Simple diversity metric - mean pairwise distance
    if n_particles > 1:
        pairwise_dists = np.linalg.norm(
            particles[:, None, :] - particles[None, :, :], axis=2
        )
        # Use upper triangle to avoid double counting
        mask = np.triu(np.ones((n_particles, n_particles)), k=1)
        diversity = np.mean(pairwise_dists * mask)
    else:
        diversity = np.array(1.0)
    
    return entropy, diversity


def safe_append_history(history_array, new_value):
    """Safely append to history array, handling None and empty cases"""
    if history_array is None:
        return np.array([new_value])
    elif history_array.size == 0:
        return np.array([new_value])
    else:
        return np.append(history_array, new_value)


def safe_get_last_value(history_array, default_value):
    """Safely get last value from history array"""
    if history_array is None or history_array.size == 0:
        return default_value
    else:
        return history_array[-1]


def t_de_particle_grad(key: jax.random.PRNGKey,
                       pid: PID,
                       target: Target,
                       particles: jax.Array,
                       y: jax.Array,
                       mc_n_samples: int,
                       lambda_r):
    """Temperature-annealed particle gradient computation"""
    def ediff_score(particle, eps):
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        assert samples.shape == (mc_n_samples, target.dim)
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        assert logp.shape == (mc_n_samples,)
        assert logq.shape == (mc_n_samples,)
        logp = np.mean(logp, 0)
        logq = np.mean(logq, 0)
        return logq - logp
    
    eps = pid.conditional.base_sample(key, mc_n_samples)
    grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
    
    # Add temperature-dependent regularization
    regularization_grad = lambda_r * particles
    
    return grad + regularization_grad


def t_de_particle_step(key: jax.random.PRNGKey,
                       pid: PID,
                       target: Target,
                       y: jax.Array,
                       optim: PIDOpt,
                       carry: TPIDCarry,
                       hyperparams: TPIDParameters):
    """Temperature-annealed particle step - safe indexing"""
    
    # Compute lambda_r 
    lambda_r = compute_lambda_r(hyperparams, carry.iteration)
    
    # Compute particle diagnostics
    entropy, diversity = compute_particle_diagnostics(pid.particles)
    
    # Simple stopping condition for now (avoid complex boolean logic)
    stop_annealing = carry.annealing_stopped
    
    def grad_fn(particles):
        return t_de_particle_grad(
            key, pid, target, particles, y, hyperparams.mc_n_samples, lambda_r
        )
    
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles, grad_fn, carry.r_precon_state
    )
    
    # Temperature-dependent noise
    noise_key, _ = jax.random.split(key)
    noise = jax.random.normal(noise_key, pid.particles.shape) * np.sqrt(2 * lambda_r)
    
    update, r_opt_state = optim.r_optim.update(
        g_grad, carry.r_opt_state, params=pid.particles, index=y
    )
    
    # Add temperature-dependent noise to the update
    update = update + noise
    
    pid = eqx.tree_at(lambda tree: tree.particles, pid, pid.particles + update)
    
    # Update history arrays using safe functions
    new_entropy_history = safe_append_history(carry.entropy_history, entropy)
    new_diversity_history = safe_append_history(carry.diversity_history, diversity)
    
    new_carry = TPIDCarry(
        id=pid,
        theta_opt_state=carry.theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state,
        iteration=carry.iteration + 1,
        current_lambda_r=lambda_r,
        annealing_stopped=stop_annealing,
        entropy_history=new_entropy_history,
        diversity_history=new_diversity_history
    )
    
    return pid, new_carry


def t_de_loss(key: jax.random.PRNGKey,
              params: PyTree,
              static: PyTree,
              target: Target,
              y: jax.Array,
              hyperparams: TPIDParameters):
    """Temperature-annealed density estimation loss"""
    pid = eqx.combine(params, static)
    _samples = pid.sample(key, hyperparams.mc_n_samples, None)
    logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
    logp = vmap(target.log_prob, (0, None))(_samples, y)
    return np.mean(logq - logp, axis=0)


def t_de_step(key: jax.random.PRNGKey,
              carry: TPIDCarry,
              target: Target,
              y: jax.Array,
              optim: PIDOpt,
              hyperparams: TPIDParameters):
    """Temperature-annealed density estimation step"""
    theta_key, r_key = jax.random.split(key, 2)
    
    def loss(key, params, static):
        return t_de_loss(key, params, static, target, y, hyperparams)
    
    lval, pid, theta_opt_state = loss_step(
        theta_key, loss, carry.id, optim.theta_optim, carry.theta_opt_state
    )
    
    # Update carry with new theta_opt_state
    carry = TPIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state,
        iteration=carry.iteration,
        current_lambda_r=carry.current_lambda_r,
        annealing_stopped=carry.annealing_stopped,
        entropy_history=carry.entropy_history,
        diversity_history=carry.diversity_history
    )
    
    pid, carry = t_de_particle_step(r_key, pid, target, y, optim, carry, hyperparams)
    
    return lval, carry


def get_annealing_diagnostics(carry: TPIDCarry) -> dict:
    """Get current annealing diagnostics - safe version"""
    
    # Safe access to history arrays
    current_entropy = safe_get_last_value(carry.entropy_history, 0.0)
    current_diversity = safe_get_last_value(carry.diversity_history, 1.0)
    
    return {
        'iteration': carry.iteration,
        'current_lambda_r': carry.current_lambda_r,
        'annealing_stopped': carry.annealing_stopped,
        'current_entropy': current_entropy,
        'current_diversity': current_diversity,
        'entropy_history': carry.entropy_history if carry.entropy_history is not None else np.array([]),
        'diversity_history': carry.diversity_history if carry.diversity_history is not None else np.array([])
    }


def print_annealing_status(carry: TPIDCarry, verbose: bool = True):
    """Print current annealing status - safe wrapper"""
    if verbose:
        try:
            diagnostics = get_annealing_diagnostics(carry)
            iteration = int(diagnostics['iteration'])
            lambda_r = float(diagnostics['current_lambda_r'])
            entropy = float(diagnostics['current_entropy'])
            diversity = float(diagnostics['current_diversity'])
            stopped = bool(diagnostics['annealing_stopped'])
            
            print(f"Iteration {iteration}: "
                  f"λᵣ = {lambda_r:.6f}, "
                  f"Entropy = {entropy:.4f}, "
                  f"Diversity = {diversity:.4f}, "
                  f"Annealing stopped: {stopped}")
        except:
            # Skip printing if we're in a traced context or have other issues
            pass


# Initialize T-PVI carry with proper empty arrays
def init_tpvi_carry(base_carry, hyperparams: TPIDParameters):
    """Initialize T-PVI carry with proper empty arrays"""
    return TPIDCarry(
        id=base_carry.id,
        theta_opt_state=base_carry.theta_opt_state,
        r_opt_state=base_carry.r_opt_state,
        r_precon_state=base_carry.r_precon_state,
        iteration=0,
        current_lambda_r=np.array(hyperparams.lambda_0),
        annealing_stopped=False,
        entropy_history=np.array([]),  # Start with empty array
        diversity_history=np.array([])  # Start with empty array
    )