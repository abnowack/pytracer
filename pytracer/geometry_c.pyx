# cython: profile=False
import numpy as np

cimport numpy as np
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI

cpdef double solid_angle(double[:, ::1] segment, double[::1] point):
    cdef:
        double a, b, c, num, denom, angle

    a = (segment[0, 0] - point[0]) * (segment[0, 0] - point[0]) + \
        (segment[0, 1] - point[1]) * (segment[0, 1] - point[1])
    b = (segment[1, 0] - point[0]) * (segment[1, 0] - point[0]) + \
        (segment[1, 1] - point[1]) * (segment[1, 1] - point[1])
    c = (segment[0, 0] - segment[1, 0]) * (segment[0, 0] - segment[1, 0]) + \
        (segment[0, 1] - segment[1, 1]) * (segment[0, 1] - segment[1, 1])

    num = a + b - c
    denom = 2 * sqrt(a) * sqrt(b)
    angle = acos(fabs(num / denom))
    if angle > M_PI / 2.:
        angle = M_PI - angle

    return angle


cpdef void solid_angles(double[:, :, ::1] segments, double[::1] point, double[::1] rcache):
    cdef:
        int i
            
    for i in range(segments.shape[0]):
        rcache[i] = solid_angle(segments[i], point)
