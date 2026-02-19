import numpy as np

def slerp(v0, v1, num, t0=0, t1=1, DOT_THRESHOLD=0.9995):
    """
    Vectorized spherical linear interpolation (SLERP) for batches of vectors.

    Args:
        v0: array of shape (batch_size, dim)
        v1: array of shape (batch_size, dim)
        num: number of interpolation steps
        t0: start of interpolation (default 0)
        t1: end of interpolation (default 1)
        DOT_THRESHOLD: threshold to switch to linear interpolation
    Returns:
        array of shape (batch_size, num, dim)
    """
    v0_norm = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)

    # Dot product along last axis
    dot = np.sum(v0_norm * v1_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)  # Avoid numerical errors

    # Determine which pairs are close enough to use lerp
    use_lerp = np.abs(dot) > DOT_THRESHOLD

    # Compute angles
    theta_0 = np.arccos(dot)  # shape: (batch_size,)
    sin_theta_0 = np.sin(theta_0)

    # Create interpolation steps
    t = np.linspace(t0, t1, num)  # shape: (num,)
    t = t.reshape(1, num, 1)      # shape: (1, num, 1) for broadcasting

    # Reshape vectors for broadcasting
    v0_b = v0[:, None, :]  # (batch_size, 1, dim)
    v1_b = v1[:, None, :]  # (batch_size, 1, dim)

    theta_0_b = theta_0[:, None, None]        # (batch_size, 1, 1)
    sin_theta_0_b = sin_theta_0[:, None, None]  # (batch_size, 1, 1)

    # Slerp formula
    theta_t = theta_0_b * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0_b - theta_t) / sin_theta_0_b
    s1 = sin_theta_t / sin_theta_0_b

    result = s0 * v0_b + s1 * v1_b  # (batch_size, num, dim)

    # Handle lerp case where vectors are almost colinear
    if np.any(use_lerp):
        lerp = (1 - t) * v0_b + t * v1_b
        result[use_lerp] = lerp[use_lerp]

    return result
