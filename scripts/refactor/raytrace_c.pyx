cimport cython

from libc.math cimport floor, ceil
from numpy.math cimport INFINITY

cpdef double c_raytrace_siddon(double[::1] line, double ex1, double ex2, double ey1, double ey2, double[:, ::1] pixels) except *:
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
        int imax, imin, jmax, jmin
        int Np
        double eps = 1.e-12

    p1x = line[0]
    p1y = line[1]
    p2x = line[2]
    p2y = line[3]
    bx, by = ex1, ey1
    Nx, Ny = pixels.shape[0] + 1, pixels.shape[1] + 1
    dx, dy = (ex2 - ex1) / pixels.shape[0], (ey2 - ey1) / pixels.shape[1]

    if abs(p1y - p2y) < eps:
        if p1y >= (ey2 - eps) and p2y >= (ey2 - eps):
            return 0
        elif p1y <= (ey1 + eps) and p2y <= (ey1 + eps):
            return 0

    if abs(p1x - p2x) < eps:
        if p1x >= (ex2 - eps) and p2x >= (ex2 - eps):
            return 0
        elif p1x <= (ex1 + eps) and p2x <= (ex1 + eps):
            return 0

    if abs(p1x - p2x) < eps:
        alphaxmin = 0
        alphaxmax = 0
    else:
        alphaxmin = min(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))
        alphaxmax = max(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))

    if abs(p1y - p2y) < eps:
        alphaymin = 0
        alphaymax = 0
    else:
        alphaymin = min(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))
        alphaymax = max(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))

    if abs(p1x - p2x) < eps:
        alphamin = max(0, alphaymin)
        alphamax = min(1, alphaymax)
    elif abs(p1y - p2y) < eps:
        alphamin = max(0, alphaxmin)
        alphamax = min(1, alphaxmax)
    else:
        alphamin = max(0, alphaxmin, alphaymin)
        alphamax = min(1, alphaxmax, alphaymax)

    if p1x < p2x:
        if alphamin == alphaxmin:
            imin = 1
        else:
            imin = <int>ceil(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imax = Nx - 1
        else:
            imax = <int>floor(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if abs(p1x - p2x) < eps:
            alphax = INFINITY
        else:
            alphax = ((bx + imin * dx) - p1x) / (p2x - p1x)

    else:
        if alphamin == alphaxmin:
            imax = Nx - 2
        else:
            imax = <int>floor(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imin = 0
        else:
            imin = <int>ceil(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if abs(p1x - p2x) < eps:
            alphax = INFINITY
        else:
            alphax = ((bx + imax * dx) - p1x) / (p2x - p1x)

    if p1y < p2y:
        if alphamin == alphaymin:
            jmin = 1
        else:
            jmin = <int>ceil(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmax = Ny - 1
        else:
            jmax = <int>floor(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if abs(p1y - p2y) < eps:
            alphay =  INFINITY
        else:
            alphay = ((by + jmin * dy) - p1y) / (p2y - p1y)

    else:
        if alphamin == alphaymin:
            jmax = Ny - 2
        else:
            jmax = <int>floor(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmin = 0
        else:
            jmin = <int>ceil(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if abs(p1y - p2y) < eps:
            alphay = INFINITY
        else:
            alphay = ((by + jmax * dy) - p1y) / (p2y - p1y)

    Np = (imax - imin + 1) + (jmax - jmin + 1)

    alphamid = (min(alphax, alphay) + alphamin) / 2

    i = <int>floor(((p1x + alphamid * (p2x - p1x)) - bx) / dx)
    j = <int>floor(((p1y + alphamid * (p2y - p1y)) - by) / dy)

    if abs(p1x - p2x) < eps:
        alphaxu = 0
    else:
        alphaxu = dx / abs(p2x - p1x)
    if abs(p1y - p2y) < eps:
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

    # print(iu, ju, alphaxu, alphayu, alphac)

    while (alphamax - alphac > eps) and ((alphamax - alphax > eps) and (alphamax - alphay > eps)):
        # print('-', alphax, alphay, alphac)
        if alphax < alphay and (alphax - alphac > eps):
            lij = (alphax - alphac) * dconv
            d12 = d12 + lij * pixels[i, j]
            i = i + iu
            alphac = alphax
            alphax = alphax + alphaxu
        elif (alphay - alphac > eps):
            lij = (alphay - alphac) * dconv
            # print(i, j, alphay, alphac, alphamax)
            d12 = d12 + lij * pixels[i, j]
            j = j + ju
            alphac = alphay
            alphay = alphay + alphayu
        else:
            break

    # have to think about this for case of line in and outside of image
    # alphamax == 1 means last point is in image
    # alphamin == 0 means first point is in image
    # print(alphamax, alphamin, alphaxmin, alphaxmax)
    if alphamax == 1 and alphac < alphamax:
        print(d12, alphac, alphamax, i, j, pixels.shape[0], pixels.shape[1])
        lij = (alphamax - alphac) * dconv
        d12 = d12 + lij * pixels[i, j]

    return d12

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void c_raytrace_siddon_bulk(double[:, ::1] lines, double ex1, double ex2, double ey1, double ey2, double[:, ::1] pixels, double[::1] cache):
    cdef:
        int i

    for i in range(lines.shape[0]):
        cache[i] = c_raytrace_siddon(lines[i], ex1, ex2, ey1, ey2, pixels)


cpdef int c_raytrace_siddon_store(double[::1] line, double ex1, double ex2, double ey1, double ey2, double[:, ::1] pixels,
                                int[:, ::1] pixel_cache, double[::1] distance_cache):
    # Fixed issue, alphax[0] in Filip Jacob's paper means first alphax in siddon array, not alphax at zero.
    cdef:
        double alphaxmin, alphaxmax
        double alphaymin, alphaymax
        double p1x, p1y, p2x, p2y
        double bx, by
        int Nx, Ny
        double dx, dy
        double alphax, alphay
        double alphamid
        int i, j, iu, ju
        int imax, imin, jmax, jmin
        int Np
        double eps = 1.e-12
        int pixel_index = 0

    p1x = line[0]
    p1y = line[1]
    p2x = line[2]
    p2y = line[3]
    bx, by = ex1, ey1
    Nx, Ny = pixels.shape[0] + 1, pixels.shape[1] + 1
    dx, dy = (ex2 - ex1) / pixels.shape[0], (ey2 - ey1) / pixels.shape[1]

    if abs(p1y - p2y) < eps:
        if p1y >= (ey2 - eps) and p2y >= (ey2 - eps):
            return 0
        elif p1y <= (ey1 + eps) and p2y <= (ey1 + eps):
            return 0

    if abs(p1x - p2x) < eps:
        if p1x >= (ex2 - eps) and p2x >= (ex2 - eps):
            return 0
        elif p1x <= (ex1 + eps) and p2x <= (ex1 + eps):
            return 0

    if abs(p1x - p2x) < eps:
        alphaxmin = 0
        alphaxmax = 0
    else:
        alphaxmin = min(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))
        alphaxmax = max(((bx + 0 * dx) - p1x) / (p2x - p1x),
                        ((bx + (Nx - 1) * dx) - p1x) / (p2x - p1x))

    if abs(p1y - p2y) < eps:
        alphaymin = 0
        alphaymax = 0
    else:
        alphaymin = min(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))
        alphaymax = max(((by + 0 * dy) - p1y) / (p2y - p1y),
                        ((by + (Ny - 1) * dy) - p1y) / (p2y - p1y))

    if abs(p1x - p2x) < eps:
        alphamin = max(0, alphaymin)
        alphamax = min(1, alphaymax)
    elif abs(p1y - p2y) < eps:
        alphamin = max(0, alphaxmin)
        alphamax = min(1, alphaxmax)
    else:
        alphamin = max(0, alphaxmin, alphaymin)
        alphamax = min(1, alphaxmax, alphaymax)

    if p1x < p2x:
        if alphamin == alphaxmin:
            imin = 1
        else:
            imin = <int>ceil(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imax = Nx - 1
        else:
            imax = <int>floor(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if abs(p1x - p2x) < eps:
            alphax = INFINITY
        else:
            alphax = ((bx + imin * dx) - p1x) / (p2x - p1x)

    else:
        if alphamin == alphaxmin:
            imax = Nx - 2
        else:
            imax = <int>floor(((p1x + alphamin * (p2x - p1x)) - bx) / dx)

        if alphamax == alphaxmax:
            imin = 0
        else:
            imin = <int>ceil(((p1x + alphamax * (p2x - p1x)) - bx) / dx)

        if abs(p1x - p2x) < eps:
            alphax = INFINITY
        else:
            alphax = ((bx + imax * dx) - p1x) / (p2x - p1x)

    if p1y < p2y:
        if alphamin == alphaymin:
            jmin = 1
        else:
            jmin = <int>ceil(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmax = Ny - 1
        else:
            jmax = <int>floor(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if abs(p1y - p2y) < eps:
            alphay =  INFINITY
        else:
            alphay = ((by + jmin * dy) - p1y) / (p2y - p1y)

    else:
        if alphamin == alphaymin:
            jmax = Ny - 2
        else:
            jmax = <int>floor(((p1y + alphamin * (p2y - p1y)) - by) / dy)

        if alphamax == alphaymax:
            jmin = 0
        else:
            jmin = <int>ceil(((p1y + alphamax * (p2y - p1y)) - by) / dy)

        if abs(p1y - p2y) < eps:
            alphay = INFINITY
        else:
            alphay = ((by + jmax * dy) - p1y) / (p2y - p1y)

    Np = (imax - imin + 1) + (jmax - jmin + 1)

    alphamid = (min(alphax, alphay) + alphamin) / 2

    i = <int>floor(((p1x + alphamid * (p2x - p1x)) - bx) / dx)
    j = <int>floor(((p1y + alphamid * (p2y - p1y)) - by) / dy)

    if abs(p1x - p2x) < eps:
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

    while (alphamax - alphac > eps) and ((alphamax - alphax > eps) and (alphamax - alphay > eps)):
        if alphax < alphay:
            lij = (alphax - alphac) * dconv
            pixel_cache[pixel_index, 0] = i
            pixel_cache[pixel_index, 1] = j
            distance_cache[pixel_index] = lij
            pixel_index = pixel_index + 1
            i = i + iu
            alphac = alphax
            alphax = alphax + alphaxu
        else:
            lij = (alphay - alphac) * dconv
            pixel_cache[pixel_index, 0] = i
            pixel_cache[pixel_index, 1] = j
            distance_cache[pixel_index] = lij
            pixel_index = pixel_index + 1
            j = j + ju
            alphac = alphay
            alphay = alphay + alphayu

    # have to think about this for case of line in and outside of image
    # alphamax == 1 means last point is in image
    # alphamin == 0 means first point is in image
    # print(alphamax, alphamin, alphaxmin, alphaxmax)
    if alphamax == 1 and alphac < alphamax:

        lij = (alphamax - alphac) * dconv
        pixel_cache[pixel_index, 0] = i
        pixel_cache[pixel_index, 1] = j
        distance_cache[pixel_index] = lij
        pixel_index = pixel_index + 1

    return pixel_index