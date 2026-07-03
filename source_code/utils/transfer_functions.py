"""S-shaped and V-shaped transfer functions for binarizing continuous positions."""

import numpy as np


def sigmoid(x):
    """Standard sigmoid function (numerically stable)."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def adaptive_sigmoid(x, gamma):
    """S-shaped transfer function 1 / (1 + exp(-gamma * x)); larger gamma
    means a steeper slope."""
    return sigmoid(gamma * x)


def get_adaptive_gamma(t, T, gamma_min=2.0, gamma_max=8.0):
    """Slope schedule: interpolate linearly from gamma_min (early, soft
    binarization) to gamma_max (late, near-deterministic)."""
    return gamma_min + (gamma_max - gamma_min) * (t / T)


def binarize(continuous_pos, t, T, gamma_min=2.0, gamma_max=8.0):
    """Stochastic binarization with the adaptive sigmoid.

    The search space is centered at 0 and clipped to [-6, 6]: negative
    coordinates make a feature likely deselected, positive ones likely
    selected, and x = 0 is a coin flip.
    """
    gamma = get_adaptive_gamma(t, T, gamma_min, gamma_max)
    prob = adaptive_sigmoid(continuous_pos, gamma)
    rand = np.random.random(continuous_pos.shape)
    return (rand < prob).astype(int)


def v_shaped(x):
    """V-shaped transfer function: |tanh(x/2)|."""
    return np.abs(np.tanh(x / 2.0))


def binarize_v_shaped(continuous_pos):
    """Binarize using V-shaped transfer function."""
    prob = v_shaped(continuous_pos)
    rand = np.random.random(continuous_pos.shape)
    binary = continuous_pos.copy()
    flip_mask = rand < prob
    binary[flip_mask] = 1 - binary[flip_mask]
    return (binary > 0.5).astype(int)
