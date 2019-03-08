cimport cython

from libc.math cimport floor, ceil, sqrt
from numpy.math cimport INFINITY


cpdef double c_bilinear_interpolation(double x, double y, double[:, ::1] pixels, double ex1, double ex2, double ey1,
                                      double ey2):
    """
    NOTE: ASSUMES PIXELS IS ZERO PADDED
    a ---- b
    | x    |
    |      |
    c ---- d
    """
    cdef:
        double delx
        double dely
        int i1, j1, i2, j2
        double x1, y1
        double t, u
        double interp

    delx = (ex2 - ex1) / pixels.shape[1]
    dely = (ey2 - ey1) / pixels.shape[0]

    if x < (ex1 + delx / 2.) or x >= (ex2 - delx / 2.):
        return 0
    if y < (ey1 + dely / 2.) or y >= (ey2 - dely / 2.):
        return 0

    # get index of lower left corner
    i1 = int(floor((x - ex1 - delx / 2.) / (ex2 - ex1 - delx) * (pixels.shape[1] - 1)))
    j1 = int(floor((y - ey1 - dely / 2.) / (ey2 - ey1 - dely) * (pixels.shape[0] - 1)))
    i2 = i1 + 1
    j2 = j1 + 1

    x1 = ex1 + delx / 2. + i1 * delx
    y1 = ey1 + dely / 2. + j1 * dely

    t = (x - x1) / delx
    u = (y - y1) / dely

    interp = (1 - t) * (1 - u) * pixels[j1, i1] + \
        t * (1 - u) * pixels[j1, i2] + \
        t * u * pixels[j2, i2] + \
        (1 - t) * u * pixels[j2, i1]

    return interp


cpdef double c_raytrace_bilinear(double[::1] line, double ex1, double ex2, double ey1, double ey2, double[:, ::1] pixels,
                                double step_size=1e-3):
    # NOTE: pixels MUST be zero padded!
    # will have innacurate results otherwise
    cdef:
        double line_distance
        double bli_start, bli_end
        double integral
        int n_steps
        double step
        double bli_prev, bli_next
        int i
        double pos_x, pos_y

    line_distance = sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)

    if line_distance == 0:
        return 0.

    bli_start = c_bilinear_interpolation(line[0], line[1], pixels, ex1, ex2, ey1, ey2)
    bli_end = c_bilinear_interpolation(line[2], line[3], pixels, ex1, ex2, ey1, ey2)

    if line_distance < 2 * step_size:
        return (bli_start + bli_end) / 2 * line_distance

    integral = 0
    n_steps = int(floor(line_distance / step_size))
    step = line_distance / n_steps

    bli_prev = bli_start
    bli_next = 0.

    for i in range(n_steps - 1):
        pos_x = line[0] + (i+1) * (line[2] - line[0]) / n_steps
        pos_y = line[1] + (i+1) * (line[3] - line[1]) / n_steps

        bli_next = c_bilinear_interpolation(pos_x, pos_y, pixels, ex1, ex2, ey1, ey2)
        integral += (bli_prev + bli_next)
        bli_prev = bli_next

    integral += (bli_prev + bli_end)

    return integral * (line_distance / n_steps / 2)