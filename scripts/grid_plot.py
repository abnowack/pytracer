import sys
import matplotlib.pyplot as plt
import numpy as np
from pytracer.transmission import *
from pytracer.grid import Grid
from .assemblies import build_shielded_geometry


if __name__ == '__main__':
    sim = build_shielded_geometry(True)
    sim.grid = Grid(20, 20, 10, 10)
    sim.rotate(10)

    sim.draw(False)
    sim.grid.draw_cell(32)
    sim.grid.draw_raster_samples(32)

    plt.show()