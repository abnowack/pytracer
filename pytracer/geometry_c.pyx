# cython: profile=False
import numpy as np

cimport numpy as np
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI

cpdef void solid_angle(double[:, :, ::1] segments, double[::1] point, double[::1] rcache):
    cdef:
        double a, b, c, num, denom, angle

    for i in range(segments.shape[0]):
        a = (segments[i, 0, 0] - point[0]) * (segments[i, 0, 0] - point[0]) + \
            (segments[i, 0, 1] - point[1]) * (segments[i, 0, 1] - point[1])
        b = (segments[i, 1, 0] - point[0]) * (segments[i, 1, 0] - point[0]) + \
            (segments[i, 1, 1] - point[1]) * (segments[i, 1, 1] - point[1])
        c = (segments[i, 0, 0] - segments[i, 1, 0]) * (segments[i, 0, 0] - segments[i, 1, 0]) + \
            (segments[i, 0, 1] - segments[i, 1, 1]) * (segments[i, 0, 1] - segments[i, 1, 1])

        num = a + b - c
        denom = 2 * sqrt(a) * sqrt(b)
        angle = acos(fabs(num / denom))
        if angle > M_PI / 2.:
            angle = M_PI - angle

        rcache[i] = angle
