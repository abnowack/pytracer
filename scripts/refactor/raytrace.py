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


def raytrace_bilinear(line, extent, pixels, debug=False):

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

    def bilinear_interpolation(x, y, extent, pixels):
        Nx = pixels.shape[0]
        Ny = pixels.shape[1]

        # get pixel of (x, y)
        pi = np.floor((x - extent[0]) / (extent[1] - extent[0]) * Nx)
        pj = np.floor((y - extent[2]) / (extent[3] - extent[2]) * Ny)



