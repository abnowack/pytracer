import numpy as np


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
