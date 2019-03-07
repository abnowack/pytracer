import matplotlib.pyplot as plt

import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_raytrace_siddon, c_raytrace_siddon_bulk, c_raytrace_siddon_store

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

    integration_line = [0, 0, 0, 0]

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


def raytrace_siddon(line, extent, pixels):
    return c_raytrace_siddon(line, extent[0], extent[1], extent[2], extent[3], pixels)


def raytrace_siddon_bulk(lines, extent, pixels):
    c_raytrace_siddon_bulk(lines, extent[0], extent[1], extent[2], extent[3], pixels, _raytrace_cache)
    return np.copy(_raytrace_cache[:np.size(lines, 0)])


def raytrace_siddon_store(line, extent, pixels):
    index = c_raytrace_siddon_store(line, extent[0], extent[1], extent[2], extent[3], pixels, _pixel_cache, _distance_cache)
    return _pixel_cache[:index, :], _distance_cache[:index]


def raytrace_joseph(line, extent, pixels, debug=False):

    import matplotlib.pyplot as plt
    if debug:
        plt.figure()
        plt.imshow(pixels, extent=extent)

    x0, y0 = line[0], line[1]
    x1, y1 = line[2], line[3]

    Nx = pixels.shape[0]
    Ny = pixels.shape[1]
    bx = (x1 - x0) / Nx
    by = (y1 - y0) / Ny

    theta = np.arctan2(x1-x0, y1-y0)
    cot_theta = 1./np.tan(theta)
    tan_theta = np.tan(theta)

    y_x = lambda x: -cot_theta * x + y0
    x_y = lambda y: -tan_theta * y + x0

    if debug:
        plt.scatter([x0, x1], [y0, y1])

    # get first and last pixels
    if point_in_box(x0, y0, *extent):
        pixel_0 = point_pixel_lookup(x0, y0, extent, Nx, Ny)
        # calculate length contribution
    else:
        pixel_0 = [-1, -1]
        # calculate point of first intersection

    if point_in_box(x1, y1, *extent):
        pixel_N = point_pixel_lookup(x1, y1, extent, Nx, Ny)
        # calculate length contribution
    else:
        pixel_N = [-1, -1]
        # calculate point of last intersection



    """
    if np.abs(np.sin(theta)) >= 1 / np.sqrt(2):
        summand = 0
        for n in range(2, N-1):
            lamb = y_x(x_n) - np.floor(y_x(x_n))
            summand += pixels[i, j] + lamb * (pixels[i, j+1] - pixels[i, j])
        return summand / np.abs(np.sin(theta))
        pass
    elif np.abs(np.cos(theta)) >= 1 / np.sqrt(2):
        pass
    else:
        raise ArithmeticError
    """

    if debug:
        plt.show()


def bilinear_interp(x, y, pixels, extent):
    """
    NOTE: ASSUMES PIXELS IS ZERO PADDED
    a ---- b
    | x    |
    |      |
    c ---- d
    """
    delx = (extent[1] - extent[0]) / pixels.shape[1]
    dely = (extent[3] - extent[2]) / pixels.shape[0]

    if x < (extent[0] + delx / 2.) or x >= (extent[1] - delx / 2.):
        return 0
    if y < (extent[2] + dely / 2.) or y >= (extent[3] - dely / 2.):
        return 0

    # get index of lower left corner
    i1 = int(np.floor((x - extent[0] - delx / 2.) / (extent[1] - extent[0] - delx) * (pixels.shape[1] - 1)))
    j1 = int(np.floor((y - extent[2] - dely / 2.) / (extent[3] - extent[2] - dely) * (pixels.shape[0] - 1)))
    i2 = i1 + 1
    j2 = j1 + 1

    x1 = extent[0] + delx / 2. + i1 * delx
    y1 = extent[2] + dely / 2. + j1 * dely

    t = (x - x1) / delx
    u = (y - y1) / dely

    interp = (1 - t) * (1 - u) * pixels[j1, i1] + \
        t * (1 - u) * pixels[j1, i2] + \
        t * u * pixels[j2, i2] + \
        (1 - t) * u * pixels[j2, i1]

    return interp


def raytrace_bilinear(line, extent, pixels, step_size=1e-3, debug=False):
    # NOTE: pixels MUST be zero padded!
    # will have innacurate results otherwise

    reduced_line = line_box_overlap_line(line, extent)
    if debug:
        print(line, reduced_line)
        plt.plot([reduced_line[0], reduced_line[2]], [reduced_line[1], reduced_line[3]])

    line_distance = np.sqrt((reduced_line[2] - reduced_line[0])**2 + (reduced_line[3] - reduced_line[1])**2)

    if line_distance == 0:
        return 0.

    bli_start = bilinear_interp(reduced_line[0], reduced_line[1], pixels, extent)
    bli_end = bilinear_interp(reduced_line[2], reduced_line[3], pixels, extent)

    if line_distance < 2 * step_size:
        return (bli_start + bli_end) / 2 * line_distance

    integral = 0
    n_steps = int(np.floor(line_distance / step_size))
    step = line_distance / n_steps

    bli_prev = bli_start
    bli_next = 0.

    if debug:
        print(reduced_line)
        plt.scatter(reduced_line[0], reduced_line[1], color='r')

    for i in range(n_steps - 1):
        t = (i+1) / n_steps
        pos_x = reduced_line[0] + t * (reduced_line[2] - reduced_line[0])
        pos_y = reduced_line[1] + t * (reduced_line[3] - reduced_line[1])

        # if debug:
        #     print(t)
        #     plt.scatter(pos_x, pos_y, color='r')

        bli_next = bilinear_interp(pos_x, pos_y, pixels, extent)
        integral += (bli_prev + bli_next)
        bli_prev = bli_next

    if debug:
        plt.scatter(reduced_line[2], reduced_line[3], color='r')

    integral += (bli_prev + bli_end)

    return integral * (line_distance / n_steps / 2)


def raytrace_bulk_bilinear(lines, extent, pixels, step_size):
    sinogram = np.zeros((lines.shape[0]), dtype=np.double)

    for i in range(lines.shape[0]):
        sinogram[i] = raytrace_bilinear(lines[i], extent, pixels, step_size)

    return sinogram






