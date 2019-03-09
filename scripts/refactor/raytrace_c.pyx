cimport cython

from libc.math cimport floor, ceil, sqrt


cpdef c_line_box_overlap_line(double[::1] line, double ex1, double ex2, double ey1, double ey2):
    cdef:
        int p1_inside = -1
        int p2_inside = -1
        double lx1, ly1, lx2, ly2
        double dl, dr, db, dt
        double tl, tr, tb, tt
        double ul, ur, ub, ut
        int n_ts = 0


    # test if first point in image
    if ex1 <= line[0] <= ex2 and ey1 <= line[1] <= ey2:
        p1_inside = 1
    else:
        p1_inside = -1

    # test if second point in image
    if ex1 <= line[2] <= ex2 and ey1 <= line[3] <= ey2:
        p2_inside = 1
    else:
        p2_inside = -1

        ts1 = 2.
    ts2 = 2.

    # left side
    dl = (line[0] - line[2]) * (ey1 - ey2) - (line[1] - line[3]) * (ex1 - ex1)
    if dl != 0:
        tl = (line[0] - ex1) * (ey1 - ey2) - (line[1] - ey1) * (ex1 - ex1)
        ul = - (line[0] - line[2]) * (line[1] - ey1) + (line[1] - line[3]) * (line[0] - ex1)
        tl /= dl
        ul /= dl
        if 0 <= ul < 1 and 0 <= tl <= 1:
            ts2 = ts1
            ts1 = tl
    # tl, ul = line_line_intersection_parametric(line[0], line[1], line[2], line[3], ex1, ey1, ex1, ey2)
    # right side
    dr = (line[0] - line[2]) * (ey1 - ey2) - (line[1] - line[3]) * (ex2 - ex2)
    if dr != 0:
        tr = (line[0] - ex2) * (ey1 - ey2) - (line[1] - ey1) * (ex2 - ex2)
        ur = - (line[0] - line[2]) * (line[1] - ey1) + (line[1] - line[3]) * (line[0] - ex2)
        tr /= dr
        ur /= dr
        if 0 <= ur < 1 and 0 <= tr <= 1:
            ts2 = ts1
            ts1 = tr
    # tr, ur = line_line_intersection_parametric(line[0], line[1], line[2], line[3], ex2, ey1, ex2, ey2)
    # bottom side
    db = (line[0] - line[2]) * (ey1 - ey1) - (line[1] - line[3]) * (ex1 - ex2)
    if db != 0:
        tb = (line[0] - ex1) * (ey1 - ey1) - (line[1] - ey1) * (ex1 - ex2)
        ub = - (line[0] - line[2]) * (line[1] - ey1) + (line[1] - line[3]) * (line[0] - ex1)
        tb /= db
        ub /= db
        if 0 <= ub < 1 and 0 <= tb <= 1:
            ts2 = ts1
            ts1 = tb
    # tb, ub = line_line_intersection_parametric(line[0], line[1], line[2], line[3], ex1, ey1, ex2, ey1)
    # top side
    dt = (line[0] - line[2]) * (ey2 - ey2) - (line[1] - line[3]) * (ex1 - ex2)
    if dt != 0:
        tt = (line[0] - ex1) * (ey2 - ey2) - (line[1] - ey2) * (ex1 - ex2)
        ut = - (line[0] - line[2]) * (line[1] - ey2) + (line[1] - line[3]) * (line[0] - ex1)
        tt /= dt
        ut /= dt
        if 0 <= ut < 1 and 0 <= tt <= 1:
            ts2 = ts1
            ts1 = tt

    if 0 <= ts1 <= 1:
        n_ts += 1
    if 0 <= ts2 <= 1:
        n_ts += 1

    lx1 = line[0]
    ly1 = line[1]
    lx2 = line[2]
    ly2 = line[3]

    if n_ts == 0 and p1_inside == 1 and p2_inside == 1:
        return
    elif n_ts == 1 and p1_inside == 1:
        line[2] = lx1 + ts1 * (lx2 - lx1)
        line[3] = ly1 + ts1 * (ly2 - ly1)
    elif n_ts == 1 and p2_inside == 1:
        line[2] = lx1
        line[3] = ly1
        line[0] = lx1 + ts1 * (lx2 - lx1)
        line[1] = ly1 + ts1 * (ly2 - ly1)
    elif n_ts == 2 and p1_inside != 1 and p2_inside != 1:
        line[0] = lx1 + ts1 * (lx2 - lx1)
        line[1] = ly1 + ts1 * (ly2 - ly1)
        line[2] = lx1 + ts2 * (lx2 - lx1)
        line[3] = ly1 + ts2 * (ly2 - ly1)

    return


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

    c_line_box_overlap_line(line, ex1, ex2, ey1, ey2)

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


cpdef double c_raytrace_bulk_bilinear(double[:, ::1] lines, double ex1, double ex2, double ey1, double ey2,
                                      double[:, ::1] pixels, double[::1] sinogram, double step_size=1e-3):
    cdef:
        int i

    for i in range(lines.shape[0]):
        sinogram[i] = c_raytrace_bilinear(lines[i], ex1, ex2, ey1, ey2, pixels, step_size)