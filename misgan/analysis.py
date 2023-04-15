import math
from .rdp_accountant import compute_rdp, get_privacy_spent

def noise_mult(N, batch_size, target_eps, iterations, delta=1e-5):
    """Calculates noise_multiplier given target_eps for stochastic gradient descent.

    Args:
        N (int): Total numbers of examples
        batch_size (int): Batch size
        target_eps (float): Epsilon for DP-SGD
        delta (float): Target delta

    Returns:
        float: noise_multiplier

    Example::
        >>> noise_mult(10000, 256, 1.2, 100, 1e-5)
    """
    optimal_noise = _ternary_search(lambda noise: abs(target_eps-epsilon(N, batch_size, noise, iterations, delta)), 0.1, 50, 50)
    return optimal_noise


def epsilon(N, batch_size, noise_multiplier, iterations, delta=1e-5):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        N (int): Total numbers of examples
        batch_size (int): Batch size
        noise_multiplier (float): Noise multiplier for DP-SGD
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    q = batch_size / N
    optimal_order = _ternary_search(lambda order: _apply_dp_sgd_analysis(q, noise_multiplier, iterations, [order], delta), 1, 512, 72)
    return _apply_dp_sgd_analysis(q, noise_multiplier, iterations, [optimal_order], delta)


def _apply_dp_sgd_analysis(q, sigma, iterations, orders, delta):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        q (float): Sampling probability, generally batch_size / number_of_samples
        sigma (float): Noise multiplier
        iterations (float): Number of iterations mechanism is applied
        orders (list(float)): Orders to try for finding optimal epsilon
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    rdp = compute_rdp(q, sigma, iterations, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps


def _ternary_search(f, left, right, iterations):
    """Performs a search over a closed domain [left, right] for the value which minimizes f."""
    for i in range(iterations):
        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3
        if f(left_third) < f(right_third):
            right = right_third
        else:
            left = left_third
    return (left + right) / 2

