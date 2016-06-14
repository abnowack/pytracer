# import matplotlib.pyplot as plt
import numpy as np
import pyximport;

pyximport.install(setup_args={'include_dirs': np.get_include()})

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from geometries import build_shielded_geometry
# import math2d as m2c_py
import math2d_c as m2c
import matplotlib.pyplot as plt


def angle_matrix(angle, radian=False):
    if not radian:
        angle = angle / 180. * np.pi

    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    return rot_matrix


def radon_scan_points(diameter, height, n_vertical, n_angle):
    angles = np.linspace(0., 180., n_angle + 1)[:-1]
    vertical = np.linspace(-height, height, n_vertical)
    offset = np.zeros((2, len(angles)))
    offset[0] = np.cos(np.radians(angles)) * diameter
    offset[1] = -np.sin(np.radians(angles)) * diameter

    start = np.zeros((2, len(vertical), len(angles)))
    end = np.zeros(start.shape)

    for i, rad in enumerate(np.radians(angles)):
        start[0, :, i] = np.sin(rad) * vertical
        start[1, :, i] = np.cos(rad) * vertical
        end[:, :, i] = start[:, :, i] - offset[:, None, i]
        start[:, :, i] += offset[:, None, i]

    return start, end


def main2():
    sim = build_shielded_geometry()
    segments = sim.geometry.mesh.segments
    inner_attenuation = sim.geometry.inner_attenuation
    outer_attenuation = sim.geometry.outer_attenuation
    # print segments.shape

    # sim.draw()

    start, end = np.array([-20., 1.]), np.array([30., 0.])

    # plt.plot([start[0], end[0]], [start[1], end[1]])

    # icepts, indexes = m2c.intersections(segments, start, end)
    #
    # print np.asarray(icepts)
    # print np.asarray(indexes)
    # print

    print m2c.attenuation_length(segments, np.array([start, end]), inner_attenuation, outer_attenuation, 0.0)

    # plt.show()


def main():
    sim = build_shielded_geometry()
    segments = sim.geometry.mesh.segments
    inner_attenuation = sim.geometry.inner_attenuation
    outer_attenuation = sim.geometry.outer_attenuation

    start, end = radon_scan_points(15, 12.5, 100, 100)
    radon = np.zeros((start.shape[1], start.shape[2]))

    # sim.draw()
    # h = 5
    # for i in xrange(start.shape[1]):
    #     plt.plot([start[0, i, h], end[0, i, h]], [start[1, i, h], end[1, i, h]], c='b')

    intersects_cache = np.empty((100, 2), dtype=np.double)
    indexes_cache = np.empty(100, dtype=np.int)

    for h in xrange(start.shape[2]):
        for i in xrange(start.shape[1]):
            s, e = start[:, i, h], end[:, i, h]
            radon[i, h] = m2c.attenuation_length(segments, np.array([s, e]), inner_attenuation, outer_attenuation, 0.0,
                                                 intersects_cache, indexes_cache)

    plt.figure()
    plt.imshow(radon, interpolation='none')

    plt.show()


if __name__ == '__main__':
    main()

    # import cProfile
    # cProfile.run('main()', sort='time')
