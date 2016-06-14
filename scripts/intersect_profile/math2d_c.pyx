# cython: profile=True
import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt

cdef inline double distance(double x1, double y1, double x2, double y2):
    cdef:
        double distance = 0

    distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    distance = sqrt(distance)
    return distance

@cython.cdivision(True)
@cython.boundscheck(False)
def intersections(double[:, :, ::1] segments, double[:, ::1] path, bint ray=False, int max_intersections=100):
    cdef:
        int segments_n = segments.shape[0]
        int i
        double[:, ::1] intersects = np.empty((max_intersections, 2), dtype=np.double)
        int [::1] indexes = np.empty(max_intersections, dtype=np.int)
        double r[2]
        double s[2]
        double denom
        double p[2]
        double q[2]
        double t, u
        double intercept_x, intercept_y
        int intercepts_n = 0
        double epsilon = 1e-15

    for i in range(segments_n):
        r[0] = segments[i, 1, 0] - segments[i, 0, 0]
        r[1] = segments[i, 1, 1] - segments[i, 0, 1]
        s[0] = path[1, 0] - path[0, 0]
        s[1] = path[1, 1] - path[0, 1]

        denom = r[0] * s[1] - r[1] * s[0]
        if denom == 0.:
            return intersects[:0], indexes[:0]

        p[0] = segments[i, 0, 0]
        p[1] = segments[i, 0, 1]
        q[0] = path[0, 0]
        q[1] = path[0, 1]

        t = (q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]
        t = t / denom
        u = (q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]
        u = u / denom

        intercept_x = p[0] + t * r[0]
        intercept_y = p[1] + t * r[1]

        if -epsilon < t < 1. - epsilon:
            if not ray or 0. < u <= 1.:
                intersects[intercepts_n, 0] = intercept_x
                intersects[intercepts_n, 1] = intercept_y
                indexes[intercepts_n] = i
                intercepts_n += 1

    return intersects[:intercepts_n], indexes[:intercepts_n]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double attenuation_length(double[:, :, ::1] segments, double[:] start, double[:] end,
                       double[:] inner_attenuation, double[:] outer_attenuation,
                       double universe_attenuation):
    cdef:
        double[:, ::1] intercepts
        int[::1] indexes
        double[:, ::1] path = np.array([start, end])
        double attenuation = 0
        double current_distance = 0, min_distance = 1e15
        # int closest_index
        # double[:] closest_intercept
        double dotprod = 0.0
        int i, ci

    intercepts, indexes = intersections(segments, path)
    no_segment_intercepts = False


    # If no intersection must determine what material we are within by tracing a ray
    if intercepts.shape[0] == 0:
        intercepts, indexes = intersections(segments, path, ray=True)
        no_segment_intercepts = True
    # No intersection through a ray, must be outside the object, return atten_length from universe material
    if intercepts.shape[0] == 0:
        attenuation = distance(start[0], start[1], end[0], end[1]) * universe_attenuation
        return attenuation

    for i in range(intercepts.shape[0]):
        current_distance = distance(intercepts[i, 0], intercepts[i, 1], start[0], start[1])
        if current_distance < min_distance:
            ci = i
            min_distance = current_distance

    dotprod = (start[0] - intercepts[ci, 0]) * (segments[indexes[ci], 0, 1] - segments[indexes[ci], 1, 1])
    dotprod += (start[1] - intercepts[ci, 1]) * (segments[indexes[ci], 1, 0] - segments[indexes[ci], 0, 0])
    # start_sign = np.sign(np.dot(np.subtract(start, closest_intercept), closest_normal))

    if dotprod > 0:
        attenuation = distance(start[0], start[1], end[0], end[1]) * outer_attenuation[indexes[ci]]
    else:
        attenuation = distance(start[0], start[1], end[0], end[1]) * inner_attenuation[indexes[ci]]

    # No segment intercept, so return the beginning to end atten_length
    if no_segment_intercepts:
        return attenuation

    # Had intersections, so add up all individual atten_lengths between start to end
    for i in range(intercepts.shape[0]):
        dotprod = (start[0] - intercepts[i, 0]) * (segments[indexes[i], 0, 1] - segments[indexes[i], 1, 1])
        dotprod += (start[1] - intercepts[i, 1]) * (segments[indexes[i], 1, 0] - segments[indexes[i], 0, 0])
        if dotprod > 0:
            attenuation += sqrt((intercepts[i, 0] - end[0]) ** 2. + (intercepts[i, 1] - end[1]) ** 2.) * (inner_attenuation[indexes[i]] - outer_attenuation[indexes[i]])
        else:
            attenuation -= sqrt((intercepts[i, 0] - end[0]) ** 2. + (intercepts[i, 1] - end[1]) ** 2.) * (inner_attenuation[indexes[i]] - outer_attenuation[indexes[i]])

    return attenuation
