# More optimizations
# [ ] add is not None
# [x] boundschecking
# [ ] openmpi with prange instead of range
# [ ] nogil?

cimport cython

import numpy as np
cimport numpy as np

from cython.parallel import prange

np.import_array()

cdef extern from "tomo.h":
    void _ray_box_crop(
        double crop_ray[4], double ray[4], double extent[4]);

    void _parallel_ray(
        double ray[4], double theta, double r, double l);

    void _fan_ray(
        double ray[4], double theta, double phi, double radius);

    void _parallel_detector(
        unsigned int n, double detector_points[][4],
        double theta, double dr, double l);

    void _fan_detector(
        unsigned int n, double detector_points[][4],
        double theta, double dphi, double radius);

    double _bilinear_interpolate(
        double x, double y,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4]);

    double _forward_project(
        double ray[4], double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4], double step_size) nogil;
    
    void _back_project_parallel(
        double theta, double r[], double projection[], unsigned int n,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4]) nogil;

    void _back_project_fan(
        double theta, double phi[], double radius, double projection[], unsigned int n,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4]) nogil;

    double _detect_probability(
        double point[2], 
        double *mu, unsigned int mu_nx, unsigned int mu_ny,
        double extent[4], double detector_points[][4], unsigned int n, double step_size) nogil;
    
    double _fission_forward_project(
        double ray[4], unsigned int k,
        double *mu_pixels, double *mu_f_pixels, double *p_pixels, double *detect_prob,
        double extent[4], unsigned int nx, unsigned int ny,
        double nu_dist[], unsigned int nu_dist_n, double step_size) nogil; 


@cython.boundscheck(False)
@cython.wraparound(False)
def ray_box_crop(np.ndarray ray not None, double[::1] extent not None):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray crop_ray = np.zeros_like(ray)
    cdef double[::1] crop_ray_view_1
    cdef double[:, ::1] crop_ray_view_2
    cdef double[:, :, ::1] crop_ray_view_3

    cdef int i, j, k

    if ray.ndim == 1:
        ray_view_1 = ray
        crop_ray_view_1 = crop_ray
        _ray_box_crop(&crop_ray_view_1[0], &ray_view_1[0], &extent[0])

    elif ray.ndim == 2:
        ray_view_2 = ray
        crop_ray_view_2 = crop_ray
        for i in range(crop_ray_view_2.shape[0]):
            _ray_box_crop(&crop_ray_view_2[i, 0], &ray_view_2[i, 0], &extent[0])
    
    elif ray.ndim == 3:
        ray_view_3 = ray
        crop_ray_view_3 = crop_ray
        for j in range(crop_ray_view_3.shape[0]):
            for i in range(crop_ray_view_3.shape[1]):
                _ray_box_crop(&crop_ray_view_3[j, i, 0], &ray_view_3[j, i, 0], &extent[0])

    return crop_ray

# TODO support int types also
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_ray(theta not None, r not None, double length):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray ray

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef double r_
    cdef double[::1] r_view_1

    cdef int i, j, k

    if isinstance(theta, float) and isinstance(r, float):
        ray = np.zeros([4], dtype=np.double)
        ray_view_1 = ray

        theta_ = theta
        r_ = r

        _parallel_ray(&ray_view_1[0], theta_, r_, length)
    
    elif isinstance(theta, float) and isinstance(r, np.ndarray):
        ray = np.zeros([r.shape[0], 4], dtype=np.double)
        ray_view_2 = ray

        theta_ = theta
        r_view_1 = r

        for i in range(r_view_1.shape[0]):
            _parallel_ray(&ray_view_2[i, 0], theta_, r_view_1[i], length)
    
    elif isinstance(theta, np.ndarray) and isinstance(r, np.ndarray):
        ray = np.zeros([theta.shape[0], r.shape[0], 4], dtype=np.double)
        ray_view_3 = ray

        theta_view_1 = theta
        r_view_1 = r

        for j in range(theta_view_1.shape[0]):
            for i in range(r_view_1.shape[0]):
                _parallel_ray(&ray_view_3[j, i, 0], theta_view_1[j], r_view_1[i], length)

    return ray


# TODO support int types also
@cython.boundscheck(False)
@cython.wraparound(False)
def fan_ray(theta not None, phi not None, double radius):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray ray

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef double phi_
    cdef double[::1] phi_view_1

    cdef int i, j, k

    if isinstance(theta, float) and isinstance(phi, float):
        theta_ = theta
        phi_ = phi

        ray = np.zeros([4], dtype=np.double)
        ray_view_1 = ray

        _fan_ray(&ray_view_1[0], theta_, phi_, radius)
    
    elif isinstance(theta, float) and isinstance(phi, np.ndarray):
        theta_ = theta
        phi_view_1 = phi

        ray = np.zeros([phi_view_1.shape[0], 4], dtype=np.double)
        ray_view_2 = ray

        for i in range(phi_view_1.shape[0]):
            _fan_ray(&ray_view_2[i, 0], theta_, phi_view_1[i], radius)
    
    elif isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
        theta_view_1 = theta
        phi_view_1 = phi

        ray = np.zeros([theta_view_1.shape[0], phi_view_1.shape[0], 4], dtype=np.double)
        ray_view_3 = ray

        for j in range(theta_view_1.shape[0]):
            for i in range(phi_view_1.shape[0]):
                _fan_ray(&ray_view_3[j, i, 0], theta_view_1[j], phi_view_1[i], radius)

    return ray

# TODO support int for theta
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_detector(unsigned int n, theta not None, double dr, double l):
    cdef np.ndarray detector_points

    cdef double[:, ::1] detector_points_view_1
    cdef double[:, :, ::1] detector_points_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef int i

    if isinstance(theta, float):
        theta_ = theta

        detector_points = np.zeros([n, 4], dtype=np.double)
        detector_points_view_1 = detector_points

        _parallel_detector(n, <double (*)[4]>&detector_points_view_1[0, 0], theta_, dr, l)

    elif isinstance(theta, np.ndarray):
        theta_view_1 = theta

        detector_points = np.zeros([theta_view_1.shape[0], n, 4], dtype=np.double)
        detector_points_view_2 = detector_points

        for i in range(theta_view_1.shape[0]):
            _parallel_detector(n, <double (*)[4]>&detector_points_view_2[i, 0, 0], theta_view_1[i], dr, l)

    return detector_points


# TODO support int for theta
@cython.boundscheck(False)
@cython.wraparound(False)
def fan_detector(unsigned int n, theta not None, double dphi, double radius):
    cdef np.ndarray detector_points

    cdef double[:, ::1] detector_points_view_1
    cdef double[:, :, ::1] detector_points_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef int i

    if isinstance(theta, float):
        theta_ = theta

        detector_points = np.zeros([n, 4], dtype=np.double)
        detector_points_view_1 = detector_points

        _fan_detector(n, <double (*)[4]>&detector_points_view_1[0, 0], theta_, dphi, radius)

    elif isinstance(theta, np.ndarray):
        theta_view_1 = theta

        detector_points = np.zeros([theta_view_1.shape[0], n, 4], dtype=np.double)
        detector_points_view_2 = detector_points

        for i in range(theta_view_1.shape[0]):
            _fan_detector(n, <double (*)[4]>&detector_points_view_2[i, 0, 0], theta_view_1[i], dphi, radius)

    return detector_points

@cython.boundscheck(False)
@cython.wraparound(False)
def bilinear_interpolate(x not None, y not None, double[:, ::1] pixels not None, 
    double[::1] extent not None):
    cdef double value 
    cdef x_
    cdef y_

    cdef np.ndarray values
    cdef double[::1] values_view_1
    cdef double[::1] x_view_1
    cdef double[::1] y_view_1

    cdef int i

    if isinstance(x, float) and isinstance(y, float):
        x_ = x
        y_ = y
        value = _bilinear_interpolate(x_, y_, &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])
        return value
    
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        x_view_1 = x
        y_view_1 = y
        values = np.ndarray([x_view_1.shape[0]], dtype=np.double)
        values_view_1 = values

        for i in range(x_view_1.shape[0]):
            values_view_1[i] = _bilinear_interpolate(x_view_1[i], y_view_1[i], &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])

        return values


@cython.boundscheck(False)
@cython.wraparound(False)
def forward_project(np.ndarray ray not None, double[:, ::1] mu not None, 
    double[::1] extent not None, double step_size):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef double value
    cdef np.ndarray values

    cdef double[::1] values_view_1
    cdef double[:, ::1] values_view_2

    cdef int i, j


    if ray.ndim == 1:
        ray_view_1 = ray

        value = _forward_project(&ray_view_1[0], &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0], step_size)
        return value
    
    elif ray.ndim == 2:
        ray_view_2 = ray
        values = np.zeros([ray_view_2.shape[0]], dtype=np.double)
        values_view_1 = values

        for i in range(ray_view_2.shape[0]):
            values_view_1[i] = _forward_project(&ray_view_2[i, 0], &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0], step_size)

        return values

    elif ray.ndim == 3:
        ray_view_3 = ray
        values = np.zeros([ray_view_3.shape[0], ray_view_3.shape[1]], dtype=np.double)
        values_view_2 = values

        for j in prange(ray_view_3.shape[0], nogil=True):
            for i in range(ray_view_3.shape[1]):
                values_view_2[j, i] = _forward_project(&ray_view_3[j, i, 0], &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0], step_size)

        return values


@cython.boundscheck(False)
@cython.wraparound(False)
def back_project_parallel(theta not None, double[::1] r not None, 
    np.ndarray projection not None, unsigned int nx, unsigned int ny, 
    double[::1] extent not None):
    cdef np.ndarray back_projection = np.zeros([ny, nx], dtype=np.double)
    cdef double[:, ::1] back_projection_view = back_projection

    cdef double[::1] projection_view_1
    cdef double[:, ::1] projection_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef int i

    if isinstance(theta, float):
        projection_view_1 = projection
        theta_ = theta

        _back_project_parallel(theta_, &r[0], &projection_view_1[0], projection_view_1.shape[0],
            &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection

    elif isinstance(theta, np.ndarray):
        projection_view_2 = projection
        theta_view_1 = theta

        for i in prange(theta_view_1.shape[0], nogil=True):
            _back_project_parallel(theta_view_1[i], &r[0], &projection_view_2[i, 0], projection_view_2.shape[1],
                &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection


@cython.boundscheck(False)
@cython.wraparound(False)
def back_project_fan(theta not None, double[::1] phi not None, 
    double radius, np.ndarray projection not None, 
    unsigned int nx, unsigned int ny, double[::1] extent not None):
    cdef np.ndarray back_projection = np.zeros([ny, nx], dtype=np.double)
    cdef double[:, ::1] back_projection_view = back_projection

    cdef double[::1] projection_view_1
    cdef double[:, ::1] projection_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef int i

    if isinstance(theta, float):
        projection_view_1 = projection
        theta_ = theta

        _back_project_fan(theta_, &phi[0], radius, &projection_view_1[0], projection_view_1.shape[0],
            &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection

    elif isinstance(theta, np.ndarray):
        projection_view_2 = projection
        theta_view_1 = theta

        for i in prange(theta_view_1.shape[0], nogil=True):
            _back_project_fan(theta_view_1[i], &phi[0], radius, &projection_view_2[i, 0], projection_view_2.shape[1],
                &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection


@cython.boundscheck(False)
@cython.wraparound(False)
def detect_probability(double[:, ::1] mu, double[::1] extent,
    np.ndarray detector_points, double step_size):

    cdef np.ndarray detect_probs
    cdef double[:, ::1] detect_probs_view_1
    cdef double[:, :, ::1] detect_probs_view_2

    cdef double[:, ::1] detector_points_view_1
    cdef double[:, :, ::1] detector_points_view_2

    cdef int i, j, k

    cdef double point[2]

    cdef double dx = (extent[1] - extent[0]) / mu.shape[1]
    cdef double dy = (extent[3] - extent[2]) / mu.shape[0]

    if detector_points.ndim == 2:
        detect_probs = np.zeros([mu.shape[0], mu.shape[1]], dtype=np.double)
        detect_probs_view_1 = detect_probs
        detector_points_view_1 = detector_points

        for j in range(mu.shape[0]):
            for i in range(mu.shape[1]):
                point[0] = extent[0] + (i + 0.5) * dx
                point[1] = extent[2] + (j + 0.5) * dy

                detect_probs_view_1[j, i] = _detect_probability(point, &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0],
                    <double (*)[4]>&detector_points_view_1[0, 0], detector_points_view_1.shape[0], step_size)
    
        return detect_probs

    elif detector_points.ndim == 3:
        detect_probs = np.zeros([detector_points.shape[0], mu.shape[0], mu.shape[1]], dtype=np.double)
        detect_probs_view_2 = detect_probs
        detector_points_view_2 = detector_points

        # for k in prange(detector_points_view_2.shape[0], nogil=True):
        for k in range(detector_points_view_2.shape[0]):
            for j in range(mu.shape[0]):
                for i in range(mu.shape[1]):
                    point[0] = extent[0] + (i + 0.5) * dx
                    point[1] = extent[2] + (j + 0.5) * dy

                    detect_probs_view_2[k, j, i] = _detect_probability(point, &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0],
                        <double (*)[4]>&detector_points_view_2[k, 0, 0], detector_points_view_2.shape[1], step_size)

        return detect_probs


@cython.boundscheck(False)
@cython.wraparound(False)
def fission_forward_project(np.ndarray ray not None, unsigned int k,
    double[:, ::1] mu not None, double[:, ::1] mu_f not None, double[:, ::1] p not None, 
    np.ndarray detect_prob not None, double[::1] extent not None, 
    double[::1] nu_dist, double step_size):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef double[:, ::1] detect_prob_view_1
    cdef double[:, :, ::1] detect_prob_view_2

    cdef double value
    cdef np.ndarray values

    cdef double[::1] values_view_1
    cdef double[:, ::1] values_view_2

    cdef int i, j

    if ray.ndim == 1:
        ray_view_1 = ray
        detect_prob_view_1 = detect_prob

        value = _fission_forward_project(&ray_view_1[0], k, 
            &mu[0, 0], &mu_f[0, 0], &p[0, 0], &detect_prob_view_1[0, 0], 
            &extent[0], mu.shape[1], mu.shape[0], 
            &nu_dist[0], nu_dist.shape[0], step_size)
        return value
    
    elif ray.ndim == 2:
        ray_view_2 = ray
        detect_prob_view_1 = detect_prob
        values = np.zeros([ray_view_2.shape[0]], dtype=np.double)
        values_view_1 = values

        for i in range(ray_view_2.shape[0]):
            values_view_1[i] = _fission_forward_project(&ray_view_2[i, 0], k, 
                &mu[0, 0], &mu_f[0, 0], &p[0, 0], &detect_prob_view_1[0, 0], 
                &extent[0], mu.shape[1], mu.shape[0], 
                &nu_dist[0], nu_dist.shape[0], step_size)
        return values

    elif ray.ndim == 3:
        ray_view_3 = ray
        detect_prob_view_2 = detect_prob
        values = np.zeros([ray_view_3.shape[0], ray_view_3.shape[1]], dtype=np.double)
        values_view_2 = values

        for j in prange(ray_view_3.shape[0], nogil=True):
            for i in range(ray_view_3.shape[1]):
                values_view_2[j, i] = _fission_forward_project(&ray_view_3[j, i, 0], k, 
                    &mu[0, 0], &mu_f[0, 0], &p[0, 0], &detect_prob_view_2[j, 0, 0], 
                    &extent[0], mu.shape[1], mu.shape[0], 
                    &nu_dist[0], nu_dist.shape[0], step_size)

        return values
