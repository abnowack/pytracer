# cython: profile=False
import numpy as np

cimport numpy as np
from pytracer.transmission_c cimport point_segment_distance
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI, exp, pow


# NOTE: Can't import this from transmission_c.pyx for some reason?
cdef inline double sign_line(double x, double y, double x1, double y1, double x2, double y2):
    return (x - x1) * (y1 - y2) + (y - y1) * (x2 - x1)


@cdivision(True)
@boundscheck(False)
cpdef pfuncref_at_point(double point_x, double point_y, double[:, :, ::1] segments, int[:, ::1] pfuncrefs):
    """ Based on looking at segment with smallest distance """
    cdef:
        double min_distance = 1e99
        double distance
        double is_outer
        double point_absorbance = 0
        int pfuncref_out, pfuncref_in
        int pfuncref_point = -1

    for i in range(segments.shape[0]):
        distance = point_segment_distance(point_x, point_y, segments[i, 0, 0], segments[i, 1, 0],
                                          segments[i, 0, 1], segments[i, 1, 1])
        if distance < min_distance:
            min_distance = distance
            is_outer = sign_line(point_x, point_y, segments[i, 0, 0], segments[i, 0, 1],
                                 segments[i, 1, 0], segments[i, 1, 1])

            if pfuncrefs[i, 0] > 0:
                pfuncref_in = pfuncrefs[i, 0]
            if pfuncrefs[i, 1] > 0:
                pfuncref_out = pfuncrefs[i, 1]

            if is_outer > 0:
                pfuncref_point = pfuncref_out
            else:
                pfuncref_point = pfuncref_in

    return pfuncref_point


# def p_at_point(point_x, point_y, segments, pfuncrefs, pfuncs):
#     """ Based on looking at segment with smallest distance """
#     min_distance = 1e99
#     point_absorbance = 0
#
#     for i in range(segments.shape[0]):
#         distance = point_segment_distance(point_x, point_y, segments[i, 0, 0], segments[i, 1, 0],
#                                           segments[i, 0, 1], segments[i, 1, 1])
#         if distance < min_distance:
#             min_distance = distance
#             is_outer = sign_line(point_x, point_y, segments[i, 0, 0], segments[i, 0, 1],
#                                  segments[i, 1, 0], segments[i, 1, 1])
#             in_p = 0
#             out_p = 0
#             if pfuncrefs[i, 0] > 0:
#                 in_p = pfuncs[pfuncrefs[i, 0]](point_x, point_y)
#             if pfuncrefs[i, 1] > 0:
#                 out_p = pfuncs[pfuncrefs[i, 1]](point_x, point_y)
#
#             if is_outer == 0:
#                 point_absorbance = (in_p + out_p) / 2
#             elif is_outer > 0:
#                 point_absorbance = out_p
#             else:
#                 point_absorbance = in_p
#     return point_absorbance


@cdivision(True)
# @boundscheck(False)
cpdef pfuncref_image(int[:, ::1] image, double[::1] xs, double[::1] ys,
                     double[:, :, ::1] segments, int[:, ::1] pfuncrefs):

    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            image[i, j] = pfuncref_at_point(xs[i], ys[j], segments, pfuncrefs)
