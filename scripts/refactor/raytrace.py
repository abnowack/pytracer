import matplotlib.pyplot as plt

import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_bilinear_interpolation, c_raytrace_bilinear

_raytrace_cache = np.zeros((500000), dtype=np.double)
_pixel_cache = np.zeros((10000, 2), dtype=np.int)
_distance_cache = np.zeros((10000), dtype=np.double)


def point_in_box(x, y, bx0, bx1, by0, by1):
    if bx0 <= x <= bx1 and by0 <= y <= by1:
        return True

    return False


def point_pixel_lookup(x, y, extent, Nx, Ny):
    if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
        pass
    else:
        return None, None

    i = np.floor((x - extent[0]) / (extent[1] - extent[0]) * Nx)
    j = np.floor((y - extent[2]) / (extent[3] - extent[2]) * Ny)

    return int(i), int(j)


def line_line_intersection_parametric(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None, None

    t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t /= denom
    u = - (x1 - x2) * (y1 - y3) + (y1 - y2) * (x1 - x3)
    u /= denom

    return t, u


def line_box_intersections(line, extent):
    ts = [2, 2]

    # left side
    tl, ul = line_line_intersection_parametric(line[0], line[1], line[2], line[3], extent[0], extent[2], extent[0],
                                               extent[3])
    # right side
    tr, ur = line_line_intersection_parametric(line[0], line[1], line[2], line[3], extent[1], extent[2], extent[1],
                                               extent[3])
    # bottom side
    tb, ub = line_line_intersection_parametric(line[0], line[1], line[2], line[3], extent[0], extent[2], extent[1],
                                               extent[2])
    # top side
    tt, ut = line_line_intersection_parametric(line[0], line[1], line[2], line[3], extent[0], extent[3], extent[1],
                                               extent[3])

    n_intersections = 0

    if ul is not None and 0 <= ul < 1 and 0 <= tl <= 1:
        n_intersections += 1
        ts[1] = ts[0]
        ts[0] = tl
    if ut is not None and 0 <= ut < 1 and 0 <= tt <= 1:
        n_intersections += 1
        ts[1] = ts[0]
        ts[0] = tt
    if ur is not None and 0 <= ur < 1 and 0 <= tr <= 1:
        n_intersections += 1
        ts[1] = ts[0]
        ts[0] = tr
    if ub is not None and 0 <= ub < 1 and 0 <= tb <= 1:
        n_intersections += 1
        ts[1] = ts[0]
        ts[0] = tb

    if ts[1] < ts[0]:
        ts[0], ts[1] = ts[1], ts[0]

    return n_intersections, ts


def line_box_overlap_line(line, extent):
    # test if first point in image
    if extent[0] <= line[0] <= extent[1] and extent[2] <= line[1] <= extent[3]:
        p1_inside = True
    else:
        p1_inside = False

    # test if second point in image
    if extent[0] <= line[2] <= extent[1] and extent[2] <= line[3] <= extent[3]:
        p2_inside = True
    else:
        p2_inside = False

    n_ts, ts = line_box_intersections(line, extent)

    integration_line = np.zeros(4, dtype=np.double)

    if n_ts == 0 and p1_inside and p2_inside:
        integration_line = line
    elif n_ts == 1 and p1_inside:
        t1 = ts[0]
        integration_line[0] = line[0]
        integration_line[1] = line[1]
        integration_line[2] = line[0] + t1 * (line[2] - line[0])
        integration_line[3] = line[1] + t1 * (line[3] - line[1])
    elif n_ts == 1 and p2_inside:
        t1 = ts[0]
        integration_line[0] = line[0] + t1 * (line[2] - line[0])
        integration_line[1] = line[1] + t1 * (line[3] - line[1])
        integration_line[2] = line[0]
        integration_line[3] = line[1]
    elif n_ts == 2 and not p1_inside and not p2_inside:
        t1, t2 = ts[0], ts[1]
        integration_line[0] = line[0] + t1 * (line[2] - line[0])
        integration_line[1] = line[1] + t1 * (line[3] - line[1])
        integration_line[2] = line[0] + t2 * (line[2] - line[0])
        integration_line[3] = line[1] + t2 * (line[3] - line[1])

    return integration_line


def raytrace_bilinear(line, extent, pixels, step_size=1e-3):
    # NOTE: pixels MUST be zero padded!
    # will have innacurate results otherwise

    reduced_line = line_box_overlap_line(line, extent)
    return c_raytrace_bilinear(reduced_line, extent[0], extent[1], extent[2], extent[3], pixels, step_size)


def raytrace_bulk_bilinear(lines, extent, pixels, step_size):
    sinogram = np.zeros((lines.shape[0]), dtype=np.double)

    for i in range(lines.shape[0]):
        sinogram[i] = raytrace_bilinear(lines[i], extent, pixels, step_size)

    return sinogram

