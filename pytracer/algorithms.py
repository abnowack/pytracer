import numpy as np
from scipy.ndimage.interpolation import rotate


def solve_tikhonov(measurement, response, alpha=0):
    rr = response.reshape(-1, np.size(response, 2))
    mm = measurement.reshape((-1))
    lhs = np.dot(rr.T, rr)
    rhs = np.dot(rr.T, mm)

    # Apply Tikhonov Regularization with an L2 norm
    gamma = alpha * np.identity(np.size(lhs, 0))
    lhs += np.dot(gamma.T, gamma)

    recon = np.linalg.solve(lhs, rhs)

    return recon


def solve_tikhonov_direct(measurement, response, alpha=0):
    rr = response.reshape(-1, np.size(response, 2))
    mm = measurement.reshape((-1))
    lhs = np.dot(rr.T, rr)
    rhs = np.dot(rr.T, mm)

    # Apply Tikhonov Regularization with an L2 norm
    gamma = np.identity(np.size(lhs, 0))
    lhs += alpha * np.dot(gamma.T, gamma)
    inv = np.linalg.inv(lhs)

    recon = np.dot(inv, rhs)

    return recon


def trace_lcurve(measurement, response, alphas):
    rr = response.reshape(-1, np.size(response, 1) * np.size(response, 2))
    recon_norm = np.zeros(len(alphas))
    recon_residual = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        recon = solve_tikhonov(measurement.T, response.T, alpha=alpha)
        recon_data = np.dot(recon, rr).reshape(measurement.shape)
        recon_residual[i] = np.linalg.norm(measurement - recon_data)
        recon_norm[i] = np.linalg.norm(recon)
    return recon_norm, recon_residual


def lcurve_curvature(x, y):
    xdot = x[:-1] - x[1:]
    ydot = y[:-1] - y[1:]
    xdotdot = xdot[:-1] - xdot[1:]
    ydotdot = ydot[:-1] - ydot[1:]
    xdot = (xdot[:-1] + xdot[1:]) / 2.
    ydot = (ydot[:-1] + ydot[1:]) / 2.

    num = np.abs(xdot * ydotdot - ydot * xdotdot)
    denom = np.power(xdot * xdot + ydot * ydot, 1.5)

    return np.divide(num, denom)

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
