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

# TODO Make this more safe if intersects or indexes isn't passed correctly
@cdivision(True)
@boundscheck(False)
cpdef int intersections(double[:, :, ::1] segments, double[:, ::1] path,
                    double[:, ::1] intersects, int[::1] indexes,
                    bint ray=False):
    cdef:
        int i, intercepts_n = 0
        double r[2]
        double s[2]
        double denom, t, u, epsilon = 1e-15

    for i in range(segments.shape[0]):
        r[0] = segments[i, 1, 0] - segments[i, 0, 0]
        r[1] = segments[i, 1, 1] - segments[i, 0, 1]
        s[0] = path[1, 0] - path[0, 0]
        s[1] = path[1, 1] - path[0, 1]

        denom = r[0] * s[1] - r[1] * s[0]
        if denom == 0.:
            return 0

        t = (path[0, 0] - segments[i, 0, 0]) * s[1] - (path[0, 1] - segments[i, 0, 1]) * s[0]
        t = t / denom
        u = (path[0, 0] - segments[i, 0, 0]) * r[1] - (path[0, 1] - segments[i, 0, 1]) * r[0]
        u = u / denom

        if -epsilon < t < 1. - epsilon:
            if not ray or 0. < u <= 1.:
                intersects[intercepts_n, 0] = segments[i, 0, 0] + t * r[0]
                intersects[intercepts_n, 1] = segments[i, 0, 1] + t * r[1]
                indexes[intercepts_n] = i
                intercepts_n += 1

    return intercepts_n

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cpdef double attenuation_length(double[:, :, ::1] segments, double[:, ::1] path,
                       double[::1] inner_attenuation, double[::1] outer_attenuation,
                       double universe_attenuation, double[:, ::1] intersects_cache,
                       int[::1] indexes_cache):
    cdef:
        int n_intercepts = 0
        double attenuation = 0
        double current_distance = 0, min_distance = 1e15
        double tmp, tmp2
        int i, ci

    n_intercepts = intersections(segments, path, intersects_cache, indexes_cache)

    # If no intersection must determine what material we are within by tracing a ray
    if n_intercepts == 0:
        n_intercepts = intersections(segments, path, intersects_cache, indexes_cache, ray=True)

    # No intersection through a ray, must be outside the object, return attenuation from universe material
    if n_intercepts == 0:
        attenuation = distance(path[0, 0], path[0, 1], path[1, 0], path[1, 1]) * universe_attenuation
        return attenuation

    for i in range(n_intercepts):
        current_distance = distance(intersects_cache[i, 0], intersects_cache[i, 1], path[0, 0], path[0, 1])
        if current_distance < min_distance:
            ci = i
            min_distance = current_distance

    tmp = (path[0, 0] - intersects_cache[ci, 0]) * (segments[indexes_cache[ci], 0, 1] - segments[indexes_cache[ci], 1, 1])
    tmp += (path[0, 1] - intersects_cache[ci, 1]) * (segments[indexes_cache[ci], 1, 0] - segments[indexes_cache[ci], 0, 0])

    if tmp > 0:
        attenuation = distance(path[0, 0], path[0, 1], path[1, 0], path[1, 1]) * outer_attenuation[indexes_cache[ci]]
    else:
        attenuation = distance(path[0, 0], path[0, 1], path[1, 0], path[1, 1]) * inner_attenuation[indexes_cache[ci]]

    # Had intersections, so add up all individual attenuations between start and end
    for i in range(n_intercepts):
        tmp = (path[0, 0] - intersects_cache[i, 0]) * (segments[indexes_cache[i], 0, 1] - segments[indexes_cache[i], 1, 1])
        tmp += (path[0, 1] - intersects_cache[i, 1]) * (segments[indexes_cache[i], 1, 0] - segments[indexes_cache[i], 0, 0])
        tmp2 = sqrt((intersects_cache[i, 0] - path[1, 0]) ** 2. + (intersects_cache[i, 1] - path[1, 1]) ** 2.)
        tmp2 *= (inner_attenuation[indexes_cache[i]] - outer_attenuation[indexes_cache[i]])
        if tmp > 0:
            attenuation += tmp2
        else:
            attenuation -= tmp2
    return attenuation
