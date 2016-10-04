# cython: profile=False
import numpy as np

cimport numpy as np
from pytracer.transmission_c cimport absorbance
from pytracer.geometry_c cimport solid_angle
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI, exp

cpdef double probability_detect(double[::1] position, double[:, ::1] absorbances,
                                double[:, :, ::1] segments, double[:, :, ::1] detector_segments,
                                double universe_absorption, double[:, ::1] intersect_cache,
                                int[::1] index_cache, double[::1] cache):
    cdef:
        double prob_detect = 0
        double absorb
        double exit_prob
        double prob_solid_angle
        int i

    for i in range(detector_segments.shape[0]):
        cache[0] = (detector_segments[i, 0, 0] + detector_segments[i, 1, 0]) / 2.
        cache[1] = (detector_segments[i, 0, 1] + detector_segments[i, 1, 1]) / 2.


        absorb = absorbance(position, cache[:2], segments, absorbances, universe_absorption,
                           intersect_cache, index_cache)
        exit_prob = exp(-absorb)

        prob_solid_angle = solid_angle(detector_segments[i], position) / (2 * M_PI)
        prob_detect += prob_solid_angle * exit_prob

    return prob_detect
