# src/trainers/t_pvi.py - Temperature-Annealed PVI Implementation

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
from jax.lax import map


# Temperature Annealing Schedules
def polynomial_annealing(lambda_0: float, t: int, gamma: float = 0.55) -> float:
    """Polynomial decay: lambda_r(t) = lambda_0 * (1 + t)^(-gamma)"""
    return lambda_0 * (1 + t) ** (-gamma)


def exponential_annealing(lambda_0: float, t: int, alpha: float = 0.01) -> float:
    """Exponential decay: lambda_r(t) = lambda_0 * exp(-alpha * t)"""
    return lambda_0 * np.exp(-alpha * t)


def inverse_sigmoid_annealing(lambda_0: float, t: int, t0: float = 1000, tau: float = 100) -> float:
    """Inverse sigmoid: lambda_r(t) = lambda_0 / (1 + exp((t - t0)/tau))"""
    return lambda_0 / (1 + np.exp((t - t0) / tau))


def compute_lambda_r(hyperparams: TPIDParameters, t: int) -> float:
    """Compute regularization strength at iteration t"""
    if hyperparams.annealing_schedule == "polynomial":
        lambda_r = polynomial_annealing(hyperparams.lambda_0, t, hyperparams.gamma)
    elif hyperparams.annealing_schedule == "exponential":
        lambda_r = exponential_annealing(hyperparams.lambda_0, t, hyperparams.alpha)
    elif hyperparams.annealing_schedule == "inverse_sigmoid":
        lambda_r = inverse_sigmoid_annealing(hyperparams.lambda_0, t, hyperparams.t0, hyperparams.tau)
    else:
        raise ValueError(f"Unknown annealing schedule: {hyperparams.annealing_schedule}")
    
    # Apply floor value
    return np.maximum(lambda_r, hyperparams.lambda_min)


def compute_particle_diagnostics(particles: jax.Array) -> Tuple[float, float]:
    """Compute particle entropy and diversity metrics"""
    # Estimate entropy using particle covariance
    cov = np.cov(particles.T)
    # Regularize to avoid numerical issues
    cov_reg = cov + 1e-8 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov_reg)
    entropy = 0.5 * logdet + 0.5 * particles.shape[1] * np.log(2 * np.pi * np.e)
    
    # Compute diversity as effective sample size
    # Based on pairwise distances between particles
    pairwise_dists = np.linalg.norm(
        particles[:, None, :] - particles[None, :, :], axis=2
    )
    mean_dist = np.mean(pairwise_dists)
    diversity = mean_dist
    
    return entropy, diversity


def should_stop_annealing(carry: TPIDCarry, 
                         hyperparams: TPIDParameters, 
                         current_entropy: float,
                         current_diversity: float) -> bool:
    """Check if annealing should be stopped based on monitoring criteria"""
    if not hyperparams.monitor_entropy or carry.annealing_stopped:
        return carry.annealing_stopped
    
    # Check if entropy dropped too fast
    if len(carry.entropy_history) > 1:
        entropy_change = carry.entropy_history[-1] - current_entropy
        if entropy_change > hyperparams.entropy_threshold:
            return True
    
    # Check if diversity is too low
    if current_diversity < hyperparams.diversity_threshold:
        return True
    
    return False


def t_de_particle_grad(key: jax.random.PRNGKey,
                       pid: PID,
                       target: Target,
                       particles: jax.Array,
                       y: jax.Array,
                       mc_n_samples: int,
                       lambda_r: float):
    """
    Temperature-annealed particle gradient computation
    Incorporates time-varying regularization strength
    """
    def ediff_score(particle, eps):
        """
        Compute the expectation of the difference of scores 
        with temperature-modulated regularization
        """
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        assert samples.shape == (mc_n_samples, target.dim)
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        assert logp.shape == (mc_n_samples,)
        assert logq.shape == (mc_n_samples,)
        logp = np.mean(logp, 0)
        logq = np.mean(logq, 0)
        
        # Temperature-annealed score difference
        # Higher lambda_r initially allows more exploration
        return logq - logp
    
    eps = pid.conditional.base_sample(key, mc_n_samples)
    grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
    
    # Add temperature-dependent regularization to drift
    # This implements: b(.) = -grad_z delta_r E[theta, r] + lambda_r(t) grad_z log p_0(z)
    # Assuming standard Gaussian reference distribution p_0 = N(0, I)
    regularization_grad = lambda_r * particles  # grad_z log N(z; 0, I) = -z
    
    return grad + regularization_grad


def t_de_particle_step(key: jax.random.PRNGKey,
                       pid: PID,
                       target: Target,
                       y: jax.Array,
                       optim: PIDOpt,
                       carry: TPIDCarry,
                       hyperparams: TPIDParameters):
    """
    Temperature-annealed particle step for density estimation
    """
    # Compute current lambda_r based on iteration
    if not carry.annealing_stopped:
        lambda_r = compute_lambda_r(hyperparams, carry.iteration)
    else:
        lambda_r = hyperparams.lambda_min
    
    # Compute particle diagnostics
    entropy, diversity = compute_particle_diagnostics(pid.particles)
    
    # Check if annealing should stop
    stop_annealing = should_stop_annealing(carry, hyperparams, entropy, diversity)
    
    def grad_fn(particles):
        return t_de_particle_grad(
            key,
            pid,
            target,
            particles,
            y,
            hyperparams.mc_n_samples,
            lambda_r
        )
    
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        grad_fn,
        carry.r_precon_state,
    )
    
    # Modify the update to include temperature-dependent noise
    # This implements the diffusion term: sqrt(2 * lambda_r(t)) * dW_t
    noise_key, _ = jax.random.split(key)
    noise = jax.random.normal(noise_key, pid.particles.shape) * np.sqrt(2 * lambda_r)
    
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y
    )
    
    # Add temperature-dependent noise to the update
    update = update + noise
    
    pid = eqx.tree_at(lambda tree: tree.particles,
                      pid,
                      pid.particles + update)
    
    # Update carry with new state
    new_entropy_history = np.append(carry.entropy_history, entropy) if carry.entropy_history is not None else np.array([entropy])
    new_diversity_history = np.append(carry.diversity_history, diversity) if carry.diversity_history is not None else np.array([diversity])
    
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
    """
    Temperature-annealed density estimation loss
    """
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
              hyperparams: TPIDParameters) -> Tuple[float, TPIDCarry]:
    """
    Temperature-annealed density estimation step
    """
    theta_key, r_key = jax.random.split(key, 2)
    
    def loss(key, params, static):
        return t_de_loss(key,
                         params,
                         static,
                         target,
                         y,
                         hyperparams)
    
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
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
    
    pid, carry = t_de_particle_step(
        r_key,
        pid,
        target,
        y,
        optim,
        carry,
        hyperparams
    )
    
    return lval, carry


# Diagnostic and monitoring functions
def get_annealing_diagnostics(carry: TPIDCarry) -> dict:
    """Get current annealing diagnostics"""
    diagnostics = {
        'iteration': carry.iteration,
        'current_lambda_r': carry.current_lambda_r,
        'annealing_stopped': carry.annealing_stopped,
        'current_entropy': carry.entropy_history[-1] if carry.entropy_history is not None and len(carry.entropy_history) > 0 else 0.0,
        'current_diversity': carry.diversity_history[-1] if carry.diversity_history is not None and len(carry.diversity_history) > 0 else 0.0,
        'entropy_history': carry.entropy_history if carry.entropy_history is not None else np.array([]),
        'diversity_history': carry.diversity_history if carry.diversity_history is not None else np.array([])
    }
    return diagnostics


def print_annealing_status(carry: TPIDCarry, verbose: bool = True):
    """Print current annealing status"""
    if verbose:
        diagnostics = get_annealing_diagnostics(carry)
        print(f"Iteration {diagnostics['iteration']}: "
              f"λᵣ = {diagnostics['current_lambda_r']:.6f}, "
              f"Entropy = {diagnostics['current_entropy']:.4f}, "
              f"Diversity = {diagnostics['current_diversity']:.4f}, "
              f"Annealing stopped: {diagnostics['annealing_stopped']}")