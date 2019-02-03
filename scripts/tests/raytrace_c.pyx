from libc.math cimport ceil, floor
import numpy as np

cimport numpy as np


cpdef double raytrace_fast(double[::1] line, extent, double[:, ::1] pixels):
    # Fixed issue, alphax[0] in Filip Jacob's paper means first alphax in siddon array, not alphax at zero.
    cdef:
        double d12 = 0
        double alphaxmin, alphaxmax
        double alphaymin, alphaymax
        double p1x, p1y, p2x, p2y
        double bx, by
        int Nx, Ny
        double dx, dy
        double alphax, alphay
        double alphamid
        int i, j, iu, ju

    p1x = line[0]
    p1y = line[1]
    p2x = line[2]
    p2y = line[3]
    bx, by = extent[0], extent[2]
    Nx, Ny = pixels.shape[0] + 1, pixels.shape[1] + 1
    dx, dy = (extent[1] - extent[0]) / pixels.shape[0], (extent[3] - extent[2]) / pixels.shape[1]

    if p1x == p2x:
        alphaxmin = 0
        alphaxmax = 0
    else:
        alphaxmin = min(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))
        alphaxmax = max(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))

    if p1y == p2y:
        alphaymin = 0
        alphaymax = 0
    else:
        alphaymin = min(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))
        alphaymax = max(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))

    if p1x == p2x:
        alphamin = max(0, alphaymin)
        alphamax = min(1, alphaymax)
    elif p1y == p2y:
        alphamin = max(0, alphaxmin)
        alphamax = min(1, alphaxmax)
    else:
        alphamin = max(0, alphaxmin, alphaymin)
        alphamax = min(1, alphaxmax, alphaymax)

    if p1x < p2x:
        if alphamin == alphaxmin:
            imin = 1
        else:
            imin = ceil(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imax = Nx - 1
        else:
            imax = floor(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if p1x == p2x:
            alphax = np.inf
        else:
            alphax = ((bx + imin * dx) - p1x) / (p2x - p1x)

    else:
        if alphamin == alphaxmin:
            imax = Nx - 2
        else:
            imax = floor(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imin = 0
        else:
            imin = ceil(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if p1x == p2x:
            alphax = np.inf
        else:
            alphax = ((bx + imax * dx) - p1x) / (p2x - p1x)

    if p1y < p2y:
        if alphamin == alphaymin:
            jmin = 1
        else:
            jmin = ceil(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmax = Ny - 1
        else:
            jmax = floor(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if p1y == p2y:
            alphay =  np.inf
        else:
            alphay = ((by + jmin * dy) - p1y) / (p2y - p1y)

    else:
        if alphamin == alphaymin:
            jmax = Ny - 2
        else:
            jmax = floor(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmin = 0
        else:
            jmin = ceil(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if p1y == p2y:
            alphay = np.inf
        else:
            alphay = ((by + jmax * dy) - p1y) / (p2y - p1y)

    Np = (imax - imin + 1) + (jmax - jmin + 1)

    alphamid = (min(alphax, alphay) + alphamin) / 2

    i = <int>floor(((p1x + alphamid * (p2x - p1x)) - bx) / dx)
    j = <int>floor(((p1y + alphamid * (p2y - p1y)) - by) / dy)

    if p1x == p2x:
        alphaxu = 0
    else:
        alphaxu = dx / abs(p2x - p1x)
    if p1y == p2y:
        alphayu = 0
    else:
        alphayu = dy / abs(p2y - p1y)

    dconv = ((p2x - p1x)**2 + (p2y - p1y)**2)**0.5
    alphac = alphamin

    if p1x < p2x:
        iu = 1
    else:
        iu = -1

    if p1y < p2y:
        ju = 1
    else:
        ju = -1

    for k in range(Np):

        if alphax < alphay:
            lij = (alphax - alphac) * dconv
            d12 = d12 + lij * pixels[i, j]
            i = i + iu
            alphac = alphax
            alphax = alphax + alphaxu
        else:
            lij = (alphay - alphac) * dconv
            d12 = d12 + lij * pixels[i, j]
            j = j + ju
            alphac = alphay
            alphay = alphay + alphayu

    # have to think about this for case of line in and outside of image
    # alphamax == 1 means last point is in image
    # alphamin == 0 means first point is in image
    # print(alphamax, alphamin, alphaxmin, alphaxmax)
    if alphamax == 1:

        lij = (alphamax - alphac) * dconv
        d12 = d12 + lij * pixels[i, j]

    return d12