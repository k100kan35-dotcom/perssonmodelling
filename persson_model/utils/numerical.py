"""Numerical utilities for integration and spacing."""

import numpy as np
from typing import Callable, Tuple


def log_space(start: float, stop: float, num: int = 50, base: float = 10.0) -> np.ndarray:
    """
    Generate logarithmically spaced array.

    Parameters
    ----------
    start : float
        Starting value (not log)
    stop : float
        Ending value (not log)
    num : int, optional
        Number of points
    base : float, optional
        Logarithm base

    Returns
    -------
    np.ndarray
        Logarithmically spaced values
    """
    if start <= 0 or stop <= 0:
        raise ValueError("start and stop must be positive for log spacing")

    log_start = np.log(start) / np.log(base)
    log_stop = np.log(stop) / np.log(base)

    return np.logspace(log_start, log_stop, num, base=base)


def adaptive_integration(
    func: Callable,
    x_range: np.ndarray,
    tol: float = 1e-6,
    max_refine: int = 3
) -> float:
    """
    Adaptive numerical integration using refinement.

    Parameters
    ----------
    func : callable
        Function to integrate
    x_range : np.ndarray
        Integration points
    tol : float, optional
        Relative tolerance
    max_refine : int, optional
        Maximum refinement iterations

    Returns
    -------
    float
        Integral value
    """
    from scipy.integrate import trapz

    # Initial integration
    y = np.array([func(x) for x in x_range])
    integral_old = trapz(y, x_range)

    for _ in range(max_refine):
        # Refine by adding midpoints
        x_refined = np.zeros(2 * len(x_range) - 1)
        x_refined[::2] = x_range
        x_refined[1::2] = 0.5 * (x_range[:-1] + x_range[1:])

        y_refined = np.array([func(x) for x in x_refined])
        integral_new = trapz(y_refined, x_refined)

        # Check convergence
        if abs(integral_new - integral_old) / (abs(integral_new) + 1e-20) < tol:
            return integral_new

        x_range = x_refined
        integral_old = integral_new

    return integral_new
