# cython: profile=False
import numpy as np

cimport numpy as np
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI, exp, pow


cdef inline double distance(double x1, double y1, double x2, double y2):
    cdef:
        double tmp = 0

    tmp = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return sqrt(tmp)

cdef inline double sign_line(double x, double y, double x1, double y1, double x2, double y2):
    return (x - x1) * (y1 - y2) + (y - y1) * (x2 - x1)

@cdivision(True)
cpdef point_segment_distance(double px, double py, double x0, double x1, double y0, double y1):
    cdef:
        double length_sq, t, projection_x, projection_y, distance

    length_sq = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
    if length_sq <= 0:
        distance = sqrt((px - x0) * (px - x0) + (py - y0) * (py - y0))
        return distance

    t = (px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)
    t /= length_sq
    if t > 1:
        t = 1
    elif t < 0:
        t = 0

    projection_x = x0 + t * (x1 - x0)
    projection_y = y0 + t * (y1 - y0)
    distance = (px - projection_x) * (px - projection_x) + (py - projection_y) * (py - projection_y)
    distance = sqrt(distance)
    return distance


@cdivision(True)
@boundscheck(False)
cpdef int pfuncref_at_point(double point_x, double point_y, double[:, :, ::1] segments, int[:, ::1] pfuncrefs):
    """ Based on looking at segment with smallest distance """
    cdef:
        double min_distance = 1.0e19
        double distance = 2.0e19
        double is_outer
        int pfuncref_out, pfuncref_in
        int pfuncref_point = -1

    for i in range(segments.shape[0]):
        distance = point_segment_distance(point_x, point_y, segments[i, 0, 0], segments[i, 1, 0],
                                          segments[i, 0, 1], segments[i, 1, 1])
        if distance < min_distance:
            min_distance = distance
            is_outer = sign_line(point_x, point_y, segments[i, 0, 0], segments[i, 0, 1],
                                 segments[i, 1, 0], segments[i, 1, 1])

            if is_outer > 0:
                pfuncref_point = pfuncrefs[i, 1]
            else:
                pfuncref_point = pfuncrefs[i, 0]

    return pfuncref_point


@cdivision(True)
# @boundscheck(False)
cpdef void pfuncref_image(int[:, ::1] image, double[::1] xs, double[::1] ys,
                     double[:, :, ::1] segments, int[:, ::1] pfuncrefs):

    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            image[i, j] = pfuncref_at_point(xs[i], ys[j], segments, pfuncrefs)
