import matplotlib.pyplot as plt

import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_line_box_overlap_line, c_raytrace_bilinear, c_raytrace_bulk_bilinear, interp1d

_raytrace_cache = np.zeros((500000), dtype=np.double)
_pixel_cache = np.zeros((10000, 2), dtype=np.int)
_distance_cache = np.zeros((10000), dtype=np.double)


def point_image_lookup(x, y, extent, Nx, Ny):
    if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
        pass
    else:
        return None, None

    i = np.floor((x - extent[0]) / (extent[1] - extent[0]) * Nx)
    j = np.floor((y - extent[2]) / (extent[3] - extent[2]) * Ny)

    return int(i), int(j)


def line_box_overlap_line(ray, extent):
    line = np.copy(ray)
    c_line_box_overlap_line(line, extent[0], extent[1], extent[2], extent[3])
    return line


def raytrace_bilinear(ray, image, extent, step_size=1e-3):
    # NOTE: pixels MUST be zero padded!
    # will have innacurate results otherwise

    ray_copy = np.copy(ray)
    return c_raytrace_bilinear(ray_copy, extent[0], extent[1], extent[2], extent[3], image, step_size)


def raytrace_bulk_bilinear(rays, image, extent, step_size):
    sinogram = np.zeros((rays.shape[0]), dtype=np.double)
    rays_copy = np.copy(rays)

    c_raytrace_bulk_bilinear(rays_copy, extent[0], extent[1], extent[2], extent[3], image, sinogram, step_size)

    return sinogram


def raytrace_backproject(ray, value, image, extent, step_size=1e-3):
    ray_copy = np.copy(ray)
    c_raytrace_backproject(ray_copy, value, extent[0], extent[1], extent[2], extent[3], image, step_size)


def raytrace_backproject_bulk(rays, sinogram, image_shape, extent, step_size=1e-3):
    backprojection = np.zeros((image_shape[0], image_shape[1]), dtype=np.double)
    rays_copy = np.copy(rays)

    for i in range(rays_copy.shape[0]):
        c_raytrace_backproject(rays_copy[i], sinogram[i], extent[0], extent[1], extent[2], extent[3], backprojection, step_size)

    return backprojection


def interp(xs, ys, xnew, left=0., right=0.):
    ynew = np.zeros(xnew.shape[0], dtype=np.double)

    for i in range(xnew.shape[0]):
        ynew[i] = interp1d(xs, ys, xnew[i], left, right)

    return ynew