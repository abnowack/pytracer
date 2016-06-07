import sys

import matplotlib.pyplot as plt
import numpy as np
import weave

from pytracer.transmission import *
from pytracer.grid import Grid
from geometries import build_shielded_geometry

from intersect_module import intersecting_segments_c as isc

if __name__ == "__main__":
    sim = build_shielded_geometry()
    sim.detector.width = 50
    sim.detector.nbins = 100

    start = np.array([-20., -10.])
    end = np.array([20., 10.])

    # plt.figure()
    sim.draw()
    plt.tight_layout()

    plt.plot([start[0], end[0]], [start[1], end[1]])

    intercepts, indexes = sim.geometry.get_intersecting_segments(start, end)

    for intercept in intercepts:
        plt.plot(intercept[0], intercept[1], 'o', color='green')

    plt.show()
