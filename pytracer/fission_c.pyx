# cython: profile=False
import numpy as np

cimport numpy as np
from pytracer.transmission_c cimport absorbance, absorbance_at_point
from pytracer.geometry_c cimport solid_angle
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, acos, fabs, M_PI, exp, pow

cpdef unsigned int binom(unsigned int n, unsigned int k):
    cdef:
        unsigned int ans=1
        unsigned int j=1

    if k > n - k:
        k = n - k

    for j in range(1, k+1):
        if n % j == 0:
            ans *= n / j
        elif ans % j == 0:
            ans = ans / j * n
        else:
            ans = (ans * n) / j
        n -= 1

    return ans


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

cpdef double probability_segment_neutron(double[:, :, ::1] segments, double[:, ::1] absorbances,
                                         double[:, ::1] fission_segment, double[:, :, ::1] detector_segments,
                                         double universe_absorption, int num_segment_points,
                                         double[::1] source, int k, double[::1] nu_dist, double mu_fission,
                                         double[:, ::1] intersect_cache,
                                         int[::1] index_cache, double[::1] cache):
    cdef:
        int i, j
        double prob_ds, segment_probability = 0, segment_length
        double prob_in, prob_out, absorb

    segment_length = (fission_segment[0, 0] - fission_segment[1, 0]) * (fission_segment[0, 0] - fission_segment[1, 0])
    segment_length += (fission_segment[0, 1] - fission_segment[1, 1]) * (fission_segment[0, 1] - fission_segment[1, 1])
    segment_length = sqrt(segment_length)

    for i in range(1, num_segment_points + 1):
        cache[0] = fission_segment[0, 0] + (i - 0.5) * (fission_segment[1, 0] - fission_segment[0, 0]) / num_segment_points
        cache[1] = fission_segment[0, 1] + (i - 0.5) * (fission_segment[1, 1] - fission_segment[0, 1]) / num_segment_points

        absorb = absorbance(source, cache[:2], segments, absorbances, universe_absorption, intersect_cache, index_cache)
        prob_in = exp(-absorb)

        prob_detect = probability_detect(cache[:2], absorbances, segments, detector_segments, universe_absorption, intersect_cache, index_cache, cache[3:])
        prob_out = 0.
        for j in range(np.size(nu_dist)):
            prob_out += binom(j, k) * nu_dist[j] * pow(prob_detect, k) * pow(1. - prob_detect, j - k)

        segment_probability += prob_in * mu_fission * prob_out * segment_length / num_segment_points

    return segment_probability


cpdef double probability_segment_neutron_grid(double[:, :, ::1] segments, double[:, ::1] absorbances,
                                              double[:, ::1] fission_segment, double[:, :, ::1] detector_segments,
                                              double universe_absorption, int num_segment_points,
                                              double[::1] source, int k, double[::1] nu_dist,
                                              double[:, ::1] intersect_cache,
                                              int[::1] index_cache, double[::1] cache):
    cdef:
        int i, j
        double prob_ds, segment_probability = 0, segment_length
        double prob_in, prob_out, absorb, mu_fission
        double point_absorbance

    segment_length = (fission_segment[0, 0] - fission_segment[1, 0]) * (fission_segment[0, 0] - fission_segment[1, 0])
    segment_length += (fission_segment[0, 1] - fission_segment[1, 1]) * (fission_segment[0, 1] - fission_segment[1, 1])
    segment_length = sqrt(segment_length)

    for i in range(1, num_segment_points + 1):
        cache[0] = fission_segment[0, 0] + (i - 0.5) * (fission_segment[1, 0] - fission_segment[0, 0]) / num_segment_points
        cache[1] = fission_segment[0, 1] + (i - 0.5) * (fission_segment[1, 1] - fission_segment[0, 1]) / num_segment_points

        absorb = absorbance(source, cache[:2], segments, absorbances, universe_absorption, intersect_cache, index_cache)
        prob_in = exp(-absorb)

        mu_fission = absorbance_at_point(cache[0], cache[1], segments, absorbances)

        prob_detect = probability_detect(cache[:2], absorbances, segments, detector_segments, universe_absorption, intersect_cache, index_cache, cache[3:])
        prob_out = 0.
        for j in range(np.size(nu_dist)):
            prob_out += binom(j, k) * nu_dist[j] * pow(prob_detect, k) * pow(1. - prob_detect, j - k)

        segment_probability += prob_in * mu_fission * prob_out * segment_length / num_segment_points

    return segment_probability
