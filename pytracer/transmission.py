import numpy as np
from scipy.ndimage.interpolation import rotate
from itertools import izip
from detector import DetectorPlane
import math2d
from simulation import Simulation
from material import Material
from geometry import Geometry
from solid import Solid


def radon(sim, angles):
    if type(sim.detector) is not DetectorPlane:
        raise TypeError('self.detector is not DetectorPlane')

    radon = np.zeros((sim.detector.nbins, len(angles)))

    for i, angle in enumerate(angles):
        sim.rotate(angle)
        detector_points = math2d.center(sim.detector.segments)
        source_points = np.dot(detector_points, math2d.angle_matrix(180.))[::-1]
        for j, (source_point, detector_point) in enumerate(izip(detector_points, source_points)):
            radon[j, i] = sim.geometry.attenuation_length(source_point, detector_point)

    return radon


# TODO : Normalization isn't correct
def inverse_radon(radon, thetas):
    """
    Reconstruct using Filtered Back Projection.

    Weighting assumes thetas are equally spaced
    radon size must be even
    """
    pad_value = int(2 ** (np.ceil(np.log(2 * np.size(radon, 0)) / np.log(2))))
    pre_pad = int((pad_value - len(radon[:, 0])) / 2)
    post_pad = pad_value - len(radon[:, 0]) - pre_pad

    f = np.fft.fftfreq(pad_value)
    ramp_filter = 2. * np.abs(f)

    reconstruction_image = np.zeros((np.size(radon, 0), np.size(radon, 0)))

    for i, theta in enumerate(thetas):
        filtered = np.real(np.fft.ifft(np.fft.fft(np.pad(radon[:, i], (pre_pad, post_pad), 'constant', constant_values=(0, 0))) * ramp_filter))[pre_pad:-post_pad]
        back_projection = rotate(np.tile(filtered, (np.size(radon, 0), 1)), theta, reshape=False, mode='constant')
        reconstruction_image += back_projection * 2 * np.pi / len(thetas)

    return reconstruction_image


def build_transmission_response(sim, angles):
    """
    Use grid to define pixel elements

    Parameters
    ----------
    sim: Simulation
    angles

    Returns
    -------

    """

    if sim.grid is None:
        raise RuntimeError("Simulation must have a grid defined")

    response = np.zeros((sim.detector.nbins, np.size(angles), sim.grid.ncells))

    unit_material = Material(1.0, 0.0, color='black')
    vacuum_material = Material(0.0, 0.0, color='white')
    cell_geo = Geometry(universe_material=vacuum_material)

    for i in xrange(sim.grid.ncells):
        print i
        grid_square = sim.grid.create_mesh(i)
        cell_geo.solids = [Solid(grid_square, unit_material, vacuum_material)]
        cell_geo.flatten()
        for j, angle in enumerate(angles):
            sim.rotate(angle)
            for k, detector_center in enumerate(math2d.center(sim.detector.segments)):
                response[k, j, i] = cell_geo.attenuation_length(sim.source.pos, detector_center)

    return response


def scan(sim, angles):

    measurement = np.zeros((sim.detector.nbins, np.size(angles)))

    for i, angle in enumerate(angles):
        sim.rotate(angle)
        for j, detector_center in enumerate(math2d.center(sim.detector.segments)):
            measurement[j, i] = sim.geometry.attenuation_length(sim.source.pos, detector_center)

    return measurement


def recon_tikhonov(measurement, response, alpha=0.):
    rr = response.reshape(-1, np.size(response, 2))
    mm = measurement.reshape((-1))
    lhs = np.dot(rr.T, rr)
    rhs = np.dot(rr.T, mm)

    # Apply Tikhonov Regularization with an L2 norm
    gamma = alpha * np.identity(np.size(lhs, 0))
    lhs += np.dot(gamma.T, gamma)

    recon = np.linalg.solve(lhs, rhs)

    return recon

