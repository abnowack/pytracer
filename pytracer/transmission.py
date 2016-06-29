import numpy as np
from scipy.ndimage.interpolation import rotate
from . import geometry as geo
from . import transmission_c as trans_c

_intersect_cache = np.empty((100, 2), dtype=np.double)
_index_cache = np.empty(100, dtype=np.int)


def intersections(start, end, segments, intersect_cache=None, index_cache=None, ray=False):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    num_intersects = trans_c.intersections(start, end, segments, intersect_cache, index_cache, ray)

    return intersect_cache[:num_intersects], index_cache[:num_intersects]


def attenuation(start, end, segments, seg_attenuation, universe_attenuation=0.0,
                intersect_cache=None, index_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache

    return trans_c.attenuation(start, end, segments, seg_attenuation, universe_attenuation,
                               intersect_cache, index_cache)


def attenuations(start, end, segments, seg_attenuation, universe_attenuation=0.0,
                 intersect_cache=None, index_cache=None, attenuation_cache=None):
    if intersect_cache is None:
        intersect_cache = _intersect_cache
    if index_cache is None:
        index_cache = _index_cache
    if attenuation_cache is None:
        attenuation_cache = np.zeros(len(start), dtype=np.double)

    trans_c.attenuations(start, end, segments, seg_attenuation, universe_attenuation,
                         intersect_cache, index_cache, attenuation_cache)

    return attenuation_cache


def scan(flat_geom, start, end):
    atten = np.zeros((start.shape[:-1]), dtype=np.double)
    start = start.reshape(-1, start.shape[-1])
    end = end.reshape(-1, end.shape[-1])

    attenuations(start, end, flat_geom.segments, flat_geom.attenuation, 0, attenuation_cache=atten.ravel())

    return atten


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
        filtered = np.real(np.fft.ifft(
            np.fft.fft(np.pad(radon[:, i], (pre_pad, post_pad), 'constant', constant_values=(0, 0))) * ramp_filter))[
                   pre_pad:-post_pad]
        back_projection = rotate(np.tile(filtered, (np.size(radon, 0), 1)), theta, reshape=False, mode='constant')
        reconstruction_image += back_projection * 2 * np.pi / len(thetas)

    return reconstruction_image

# def build_transmission_response(sim, n_angles):
#     angles = np.linspace(0, 180, n_angles + 1)[:-1]
#
#     if sim.grid is None:
#         raise RuntimeError("Simulation must have a grid defined")
#
#     response = np.zeros((sim.detector.nbins, np.size(angles), sim.grid.ncells))
#     response_shape = response.shape
#     response = response.reshape((response.shape[0] * response.shape[1], response.shape[2]))
#
#     unit_material = Material(1, 0, color='black')
#     vacuum_material = Material(0, 0, color='white')
#     cell_geo = Geometry(universe_material=vacuum_material)
#
#     # paths = np.zeros((sim.detector.nbins, len(angles), 2, 2))
#     # paths[:, :, 0, :] = sim.source.pos
#     # for i, angle in enumerate(angles):
#     #     sim.rotate(angle)
#     #     paths[:, i, 1, :] = math2d.center(sim.detector.segments)
#     paths = np.zeros((sim.detector.nbins, len(angles), 2, 2))
#     for i, angle in enumerate(angles):
#         sim.rotate(angle)
#         paths[:, i, 0, :] = math2d.center(sim.detector.segments)
#         paths[:, i, 1, :] = -paths[:, i, 0, :][::-1]
#     paths = paths.reshape((paths.shape[0] * paths.shape[1], paths.shape[2], paths.shape[3]))
#
#     for i in range(sim.grid.ncells):
#         print
#         i
#         grid_square = sim.grid.create_mesh(i)
#         cell_geo.solids = [Solid(grid_square, unit_material, vacuum_material)]
#         cell_geo.flatten()
#
#         cell_geo.attenuation_length(paths, attenuation_cache=response[:, i])
#         # for j in xrange(response.shape[0]):
#         #     start, end = paths[j, 0, :], paths[j, 1, :]
#         #     response[j, i] = cell_geo.attenuation_length(start, end)
#
#     print
#
#     return response.reshape(response_shape)
