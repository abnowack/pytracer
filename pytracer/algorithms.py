import numpy as np
from scipy.ndimage.interpolation import rotate


def generalized_cross_validation(d, G, alpha):
    m = np.size(G, 0)
    L = np.identity(np.size(G, 1))

    GPound = np.linalg.inv(G.T @ G + (alpha ** 2) * L.T @ L) @ G.T
    m_alpha = GPound @ d

    num = m * np.linalg.norm(G @ m_alpha - d) ** 2
    denom = np.trace(np.identity(m) - G @ GPound) ** 2

    gcv_value = num / denom

    return gcv_value


def solve_tikhonov(d, G, alpha=0):
    gamma = alpha * np.identity(np.size(G, 1))

    lhs = G.T @ G + gamma.T @ gamma
    rhs = G.T @ d
    inv = np.linalg.inv(lhs)
    m = inv @ rhs

    return m


def trace_lcurve(d, G, alphas):
    norms = np.zeros(len(alphas))
    residuals = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        m_alpha = solve_tikhonov(d, G, alpha=alpha)
        residuals[i] = np.linalg.norm(G @ m_alpha - d)
        norms[i] = np.linalg.norm(m_alpha)
    return norms, residuals


def diff_central(x, y):
    """Calculate central derivative, return derivative values excluding endpoints."""
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1) / (x2 - x0)
    return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)


def lcurve_curvature(alphas, norms, residuals):
    eta_hat = np.log(norms ** 2)
    rho_hat = np.log(residuals ** 2)

    d_eta_hat = diff_central(alphas, eta_hat)
    d_rho_hat = diff_central(alphas, rho_hat)

    dd_eta_hat = diff_central(alphas[1:-1], d_eta_hat)
    dd_rho_hat = diff_central(alphas[1:-1], d_rho_hat)

    d_eta_hat = d_eta_hat[1:-1]
    d_rho_hat = d_rho_hat[1:-1]
    alphas = alphas[2:-2]

    curvature = 2 * (d_rho_hat * dd_eta_hat - dd_rho_hat * d_eta_hat) / (d_rho_hat ** 2 + d_eta_hat ** 2) ** (3 / 2)

    return alphas, curvature


def filtered_back_projection(sinogram, radians):
    """
    Reconstruct using Filtered Back Projection.

    Weighting assumes radians are equally spaced
    radon size must be even
    """
    pad_value = int(2 ** (np.ceil(np.log(2 * np.size(sinogram, 0)) / np.log(2))))
    pre_pad = int((pad_value - len(sinogram[:, 0])) / 2)
    post_pad = pad_value - len(sinogram[:, 0]) - pre_pad

    f = np.fft.fftfreq(pad_value)
    ramp_filter = 2. * np.abs(f)

    reconstruction_image = np.zeros((np.size(sinogram, 0), np.size(sinogram, 0)))

    for i, radian in enumerate(radians):
        filtered = np.real(np.fft.ifft(
            np.fft.fft(np.pad(sinogram[:, i], (pre_pad, post_pad), 'constant', constant_values=(0, 0))) * ramp_filter))[
                   pre_pad:-post_pad]
        back_projection = rotate(np.tile(filtered, (np.size(sinogram, 0), 1)), sinogram, reshape=False, mode='constant')
        reconstruction_image += back_projection * 2 * np.pi / len(radians)

    return reconstruction_image
