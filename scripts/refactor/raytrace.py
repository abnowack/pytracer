import matplotlib.pyplot as plt

import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_raytrace_bilinear, c_raytrace_bulk_bilinear, c_raytrace_backproject

_raytrace_cache = np.zeros((500000), dtype=np.double)
_pixel_cache = np.zeros((10000, 2), dtype=np.int)
_distance_cache = np.zeros((10000), dtype=np.double)


def point_pixel_lookup(x, y, extent, Nx, Ny):
    if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
        pass
    else:
        return None, None

    i = np.floor((x - extent[0]) / (extent[1] - extent[0]) * Nx)
    j = np.floor((y - extent[2]) / (extent[3] - extent[2]) * Ny)

    return int(i), int(j)


def raytrace_bilinear(line, extent, pixels, step_size=1e-3):
    # NOTE: pixels MUST be zero padded!
    # will have innacurate results otherwise

    return c_raytrace_bilinear(line, extent[0], extent[1], extent[2], extent[3], pixels, step_size)


def raytrace_bulk_bilinear(lines, extent, pixels, step_size):
    sinogram = np.zeros((lines.shape[0]), dtype=np.double)

    c_raytrace_bulk_bilinear(lines, extent[0], extent[1], extent[2], extent[3], pixels, sinogram, step_size)

    return sinogram


def raytrace_backproject(line, value, extent, pixels, step_size=1e-3):
    c_raytrace_backproject(line, value, extent[0], extent[1], extent[2], extent[3], pixels, step_size)


def raytrace_backproject_bulk(lines, values, extent, pixels, step_size=1e-3):
    for i in range(lines.shape[0]):
        c_raytrace_backproject(lines[i], values[i], extent[0], extent[1], extent[2], extent[3], pixels, step_size)
