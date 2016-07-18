# cython: profile=False
import numpy as np

cimport numpy as np
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt

cdef inline double distance(double x1, double y1, double x2, double y2):
    cdef:
        double tmp = 0

    tmp = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return sqrt(tmp)

cdef inline double sign_line(double x, double y, double x1, double y1, double x2, double y2):
    return (x - x1) * (y1 - y2) - (y - y1) * (x1 - x2)

# TODO Make this more safe if intersects or indexes isn't passed correctly
@cdivision(True)
@boundscheck(False)
cpdef int intersections(double[::1] start, double[::1] end, double[:, :, ::1] segments,
                        double[:, ::1] intersect_cache, int[::1] index_cache, bint ray=False):
    cdef:
        int i, num_intersect = 0
        double r[2]
        double s[2]
        double denom, t, u, epsilon = 1e-15

    for i in range(segments.shape[0]):
        r[0] = segments[i, 1, 0] - segments[i, 0, 0]
        r[1] = segments[i, 1, 1] - segments[i, 0, 1]
        s[0] = end[0] - start[0]
        s[1] = end[1] - start[1]

        denom = r[0] * s[1] - r[1] * s[0]
        if denom == 0.:
            continue

        t = (start[0] - segments[i, 0, 0]) * s[1] - (start[1] - segments[i, 0, 1]) * s[0]
        t = t / denom
        u = (start[0] - segments[i, 0, 0]) * r[1] - (start[1] - segments[i, 0, 1]) * r[0]
        u = u / denom

        if -epsilon < t < 1. - epsilon:
            if (ray) or 0. < u <= 1.:
                intersect_cache[num_intersect, 0] = segments[i, 0, 0] + t * r[0]
                intersect_cache[num_intersect, 1] = segments[i, 0, 1] + t * r[1]
                index_cache[num_intersect] = i
                num_intersect += 1

    return num_intersect

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cpdef double attenuation(double[::1] start, double[::1] end,
                         double[:, :, ::1] segments, double[:, ::1] seg_attenuation,
                         double universe_attenuation, double[:, ::1] intersect_cache,
                         int[::1] index_cache):
    cdef:
        int num_intersect = 0
        double attenuation = 0
        double current_distance = 0, min_distance = 1e15
        double tmp, tmp2
        int i, ci

    num_intersect = intersections(start, end, segments, intersect_cache, index_cache)

    # If no intersection must determine what material we are within by tracing a ray
    if num_intersect == 0:
        num_intersect = intersections(start, end, segments, intersect_cache, index_cache, ray=True)

    # No intersection through a ray, must be outside the object, return attenuation from universe material
    if num_intersect == 0:
        attenuation = distance(start[0], start[1], end[0], end[1]) * universe_attenuation
        return attenuation

    for i in range(num_intersect):
        current_distance = distance(intersect_cache[i, 0], intersect_cache[i, 1], start[0], start[1])
        if current_distance < min_distance:
            ci = index_cache[i]
            min_distance = current_distance

    tmp = sign_line(start[0], start[1], segments[ci, 0, 0], segments[ci, 0, 1], segments[ci, 1, 0], segments[ci, 1, 1])

    if tmp > 0:
        attenuation = distance(start[0], start[1], end[0], end[1]) * seg_attenuation[ci, 1]
    else:
        attenuation = distance(start[0], start[1], end[0], end[1]) * seg_attenuation[ci, 0]

    # Had intersections, so add up all individual attenuations between start and end
    for i in range(num_intersect):
        ci = index_cache[i]
        tmp = sign_line(start[0], start[1], segments[ci, 0, 0], segments[ci, 0, 1], segments[ci, 1, 0], segments[ci, 1, 1])
        tmp2 = distance(intersect_cache[i, 0], intersect_cache[i, 1], end[0], end[1]) * (seg_attenuation[ci, 0] - seg_attenuation[ci, 1])
        if tmp > 0:
            attenuation += tmp2
        else:
            attenuation -= tmp2
    return attenuation

@boundscheck(False)
@wraparound(False)
cpdef void attenuations(double[:, ::1] start, double[:, ::1] end,
                        double[:, :, ::1] segments, double[:, ::1] seg_attenuation,
                        double universe_attenuation, double[:, ::1] intersects_cache,
                        int[::1] indexes_cache, double[:] attenuation_cache):
    cdef:
        int i

    for i in range(start.shape[0]):
        attenuation_cache[i] = attenuation(start[i], end[i], segments, seg_attenuation, universe_attenuation,
                                           intersects_cache, indexes_cache)
