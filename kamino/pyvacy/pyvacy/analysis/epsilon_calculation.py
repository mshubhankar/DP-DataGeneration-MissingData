from .rdp_accountant import compute_rdp, get_privacy_spent


def noise_mult(N, batch_size, target_eps, iterations, delta=1e-5, gaussian_std=[]):
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
    optimal_noise = _ternary_search(lambda noise: abs(target_eps-epsilon(N, batch_size, noise, iterations, delta, gaussian_std)), 0.1, 50, 50)
    return optimal_noise

def comp_epsilon(qs, sigmas, iterations, delta):
    optimal_order = _ternary_search(
        lambda order: _apply_analysis(qs, sigmas, iterations, delta, [order]), 1, 512, 72)

    return _apply_analysis(qs, sigmas, iterations, delta, [optimal_order])


def _apply_analysis(qs, sigmas, iterations, delta, orders):
    """
    Compute the overall privacy cost
    :param qs a list of sample ratios
    :param sigmas a list of noise scale
    :param iterations
    :param delta
    :param orders a list of order
    """

    total_rdp = 0
    for idx in range(len(qs)):
        total_rdp += compute_rdp(qs[idx], sigmas[idx], iterations[idx], orders)

    epsilon, _, _ = get_privacy_spent(orders, total_rdp, target_delta=delta)

    return epsilon


def epsilon(N, batch_size, noise_multiplier, iterations, delta=1e-5, gaussian_std=[]):
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
    optimal_order = _ternary_search(lambda order: _apply_kamino_analysis(q, noise_multiplier, iterations, [order], delta), 1, 512, 72)
    return _apply_kamino_analysis(q, noise_multiplier, iterations, [optimal_order], delta, gaussian_std)


def _apply_kamino_analysis(q, sigma, iterations, orders, delta, gaussian_std=[]):
    """Calculates epsilon for kamino
    Args:
        q (float): Sampling probability, generally batch_size / number_of_samples
        sigma (float): Noise multiplier
        gaussian_sigma(list): Std dev. for gaussian noise additions
        iterations (float): Number of iterations mechanism is applied
        orders (list(float)): Orders to try for finding optimal epsilon
        delta (float): Target delta
    """
    total_rdp = compute_rdp(q, sigma, iterations, orders)
    for gaussian_sigma in gaussian_std:
        total_rdp += compute_rdp(1, gaussian_sigma, 1, orders)
    eps, _, opt_order = get_privacy_spent(orders, total_rdp, target_delta=delta)
    return eps


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

