import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from raytrace_c import c_raytrace_fast, c_raytrace_fast_bulk

_cache = np.zeros((500000), dtype=np.double)


def raytrace_fast(line, extent, pixels):
    return c_raytrace_fast(line, extent[0], extent[1], extent[2], extent[3], pixels)


def raytrace_bulk_fast(lines, extent, pixels):
    c_raytrace_fast_bulk(lines, extent[0], extent[1], extent[2], extent[3], pixels, _cache)
    return np.copy(_cache[:np.size(lines, 0)])

# TODO: REMOVE WHEN NOT NEEDED
def raytrace_slow(line, extent, pixels, debug=False, display_pixels=False):
    # Fixed issue, alphax[0] in Filip Jacob's paper means first alphax in siddon array, not alphax at zero.

    from math import ceil, floor

    p1x, p1y, p2x, p2y = line
    bx, by, = extent[0], extent[2]
    Nx, Ny = np.size(pixels, 0) + 1, np.size(pixels, 1) + 1
    dx, dy = (extent[1] - extent[0]) / np.size(pixels, 0), (extent[3] - extent[2]) / np.size(pixels, 1)

    p12x = lambda a_: p1x + a_ * (p2x - p1x)
    p12y = lambda a_: p1y + a_ * (p2y - p1y)

    alphax = lambda i_: ((bx + i_ * dx) - p1x) / (p2x - p1x)
    alphay = lambda j_: ((by + j_ * dy) - p1y) / (p2y - p1y)

    if abs(p1y - p2y) < 1e-12:
        if p1y >= (extent[3] - 1e-12) and p2y >= (extent[3] - 1e-12):
            return 0
        elif p1y <= (extent[2] + 1e-12) and p2y <= (extent[2] + 1e-12):
            return 0

    if abs(p1x - p2x) < 1e-12:
        if p1x >= (extent[1] - 1e-12) and p2x >= (extent[1] - 1e-12):
            return 0
        elif p1x <= (extent[0] + 1e-12) and p2x <= (extent[0] + 1e-12):
            return 0

    if p1x == p2x:
        alphaxmin = 0
        alphaxmax = 0
    else:
        alphaxmin = min(alphax(0), alphax(Nx - 1))
        alphaxmax = max(alphax(0), alphax(Nx - 1))

    if p1y == p2y:
        alphaymin = 0
        alphaymax = 0
    else:
        alphaymin = min(alphay(0), alphay(Ny - 1))
        alphaymax = max(alphay(0), alphay(Ny - 1))


    if p1x == p2x:
        alphamin = max(0, alphaymin)
        alphamax = min(1, alphaymax)
    elif p1y == p2y:
        alphamin = max(0, alphaxmin)
        alphamax = min(1, alphaxmax)
    else:
        alphamin = max(0, alphaxmin, alphaymin)
        alphamax = min(1, alphaxmax, alphaymax)

    phix = lambda a_: (p12x(a_) - bx) / dx

    if p1x < p2x:
        if alphamin == alphaxmin:
            imin = 1
        else:
            imin = ceil(phix(alphamin))

        if alphamax == alphaxmax:
            imax = Nx - 1
        else:
            imax = floor(phix(alphamax))

        if p1x == p2x:
            alphax_ = np.inf
        else:
            alphax_ = alphax(imin)

    else:
        if alphamin == alphaxmin:
            imax = Nx - 2
        else:
            imax = floor(phix(alphamin))

        if alphamax == alphaxmax:
            imin = 0
        else:
            imin = ceil(phix(alphamax))

        if p1x == p2x:
            alphax_ = np.inf
        else:
            alphax_ = alphax(imax)

    phiy = lambda a_: (p12y(a_) - by) / dy

    if p1y < p2y:
        if alphamin == alphaymin:
            jmin = 1
        else:
            jmin = ceil(phiy(alphamin))

        if alphamax == alphaymax:
            jmax = Ny - 1
        else:
            jmax = floor(phiy(alphamax))

        if p1y == p2y:
            alphay_ =  np.inf
        else:
            alphay_ = alphay(jmin)

    else:
        if alphamin == alphaymin:
            jmax = Ny - 2
        else:
            jmax = floor(phiy(alphamin))

        if alphamax == alphaymax:
            jmin = 0
        else:
            jmin = ceil(phiy(alphamax))

        if p1y == p2y:
            alphay_ = np.inf
        else:
            alphay_ = alphay(jmax)

    Np = (imax - imin + 1) + (jmax - jmin + 1)

    alphamid = (min(alphax_, alphay_) + alphamin) / 2

    i = floor(phix(alphamid))
    j = floor(phiy(alphamid))

    # if debug:
    #     draw_alpha(line, alphamin, color='blue')
    #     draw_alpha(line, alphamax, color='orange')

    if p1x == p2x:
        alphaxu = 0
    else:
        alphaxu = dx / abs(p2x - p1x)
    if p1y == p2y:
        alphayu = 0
    else:
        alphayu = dy / abs(p2y - p1y)

    d12 = 0
    dconv = ((p2x - p1x)**2 + (p2y - p1y)**2)**0.5
    alphac = alphamin

    # if debug:
    #     draw_alpha(line, alphac)

    if p1x < p2x:
        iu = 1
    else:
        iu = -1

    if p1y < p2y:
        ju = 1
    else:
        ju = -1

    while (alphamax - alphac > 1e-12):
    # for k in range(Np):
        if display_pixels:
            pixels[i, j] = 1

        if alphax_ < alphay_:
            lij = (alphax_ - alphac) * dconv
            d12 = d12 + lij * pixels[i, j]
            i = i + iu
            alphac = alphax_
            alphax_ = alphax_ + alphaxu
        else:
            lij = (alphay_ - alphac) * dconv
            d12 = d12 + lij * pixels[i, j]
            j = j + ju
            alphac = alphay_
            alphay_ = alphay_ + alphayu

        # if debug:
        #     draw_alpha(line, alphac)

    # have to think about this for case of line in and outside of image
    # alphamax == 1 means last point is in image
    # alphamin == 0 means first point is in image
    # print(alphamax, alphamin, alphaxmin, alphaxmax)
    if alphamax == 1:
        if display_pixels:
            pixels[i, j] = 1

        lij = (alphamax - alphac) * dconv
        d12 = d12 + lij * pixels[i, j]

        # if debug:
        #     draw_alpha(line, alphamax)

    # if debug:
    #     draw_algorithm(extent, pixels)
    #     draw_line(line)

    return d12


def raytrace_bulk_slow(lines, extent, pixels):
    results = np.zeros(lines.shape[0], dtype=np.double)
    for i, line in enumerate(lines):
        print(line)
        results[i] = raytrace_slow(line, extent, pixels)

    return results