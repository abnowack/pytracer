import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ..assemblies import build_shielded_geometry
from . import math2d as m2c_py
from . import math2d_c as m2c
import matplotlib.pyplot as plt


def angle_matrix(angle, radian=False):
    if not radian:
        angle = angle / 180 * np.pi

    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    return rot_matrix


def radon_scan_points(diameter, height, n_vertical, n_angle):
    angles = np.linspace(0, 180, n_angle + 1)[:-1]
    vertical = np.linspace(-height, height, n_vertical)
    offset = np.zeros((2, len(angles)))
    offset[0] = np.cos(np.radians(angles)) * diameter
    offset[1] = -np.sin(np.radians(angles)) * diameter

    paths = np.zeros((len(vertical), len(angles), 2, 2))

    for i, rad in enumerate(np.radians(angles)):
        paths[:, i, :, 0] = np.sin(rad) * vertical[:, None]
        paths[:, i, :, 1] = np.cos(rad) * vertical[:, None]
        paths[:, i, 0, :] += offset[:, i, None].T
        paths[:, i, 1, :] -= offset[:, i, None].T

    return paths


def main(plot=False, n=100):
    sim = build_shielded_geometry()
    segments = sim.geometry.mesh.segments
    inner_attenuation = sim.geometry.inner_attenuation
    outer_attenuation = sim.geometry.outer_attenuation

    paths = radon_scan_points(15, 12.5, n, n)
    radon = np.zeros((paths.shape[0] * paths.shape[1]))
    paths = paths.reshape((paths.shape[0] * paths.shape[1], paths.shape[2], paths.shape[3]))

    intersects_cache = np.empty((100, 2), dtype=np.double)
    indexes_cache = np.empty(100, dtype=np.int)
    m2c.calc_attenuation_bulk(segments, paths, inner_attenuation, outer_attenuation, 0.0,
                              intersects_cache, indexes_cache, radon)

    if plot:
        plt.figure()
        radon = radon.reshape((n, n))
        plt.imshow(radon, interpolation='none')

        plt.show()


if __name__ == '__main__':
    # main(True, n=1000)

    import cProfile

    cProfile.run('main()', sort='time')
