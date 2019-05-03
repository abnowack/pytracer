# More optimizations
# [ ] add is not None
# [ ] boundschecking
# [ ] openmpi with prange instead of range
# [ ] nogil?

import numpy as np
cimport numpy as np

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
        double extent[4], double step_size);
    
    void _back_project_parallel(
        double theta, double r[], double projection[], unsigned int n,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4]);

    void _back_project_fan(
        double theta, double phi[], double radius, double projection[], unsigned int n,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double extent[4]);


def ray_box_crop(np.ndarray ray not None, double[::1] extent not None):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray crop_ray = np.zeros_like(ray)
    cdef double[::1] crop_ray_view_1
    cdef double[:, ::1] crop_ray_view_2
    cdef double[:, :, ::1] crop_ray_view_3

    cdef unsigned int i, j, k

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
def parallel_ray(theta, r, double length):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray ray

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef double r_
    cdef double[::1] r_view_1

    cdef unsigned int i, j, k

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
def fan_ray(theta, phi, double radius):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef np.ndarray ray

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef double phi_
    cdef double[::1] phi_view_1

    cdef unsigned int i, j, k

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
def parallel_detector(unsigned int n, theta, double dr, double l):
    cdef np.ndarray detector_points

    cdef double[:, ::1] detector_points_view_1
    cdef double[:, :, ::1] detector_points_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef unsigned int i

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
def fan_detector(unsigned int n, theta, double dphi, double radius):
    cdef np.ndarray detector_points

    cdef double[:, ::1] detector_points_view_1
    cdef double[:, :, ::1] detector_points_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef unsigned int i

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


def bilinear_interpolate(x, y, double[:, ::1] pixels, double[::1] extent):
    cdef double value 
    cdef x_
    cdef y_

    cdef np.ndarray values
    cdef double[::1] values_view_1
    cdef double[::1] x_view_1
    cdef double[::1] y_view_1

    cdef unsigned int i

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


def forward_project(np.ndarray ray, double[:, ::1] mu, double[::1] extent, double step_size):
    cdef double[::1] ray_view_1
    cdef double[:, ::1] ray_view_2
    cdef double[:, :, ::1] ray_view_3

    cdef double value
    cdef np.ndarray values

    cdef double[::1] values_view_1
    cdef double[:, ::1] values_view_2

    cdef unsigned int i, j


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

        for j in range(ray_view_3.shape[0]):
            for i in range(ray_view_3.shape[1]):
                values_view_2[j, i] = _forward_project(&ray_view_3[j, i, 0], &mu[0, 0], mu.shape[1], mu.shape[0], &extent[0], step_size)

        return values

def back_project_parallel(theta, double[::1] r, np.ndarray projection, unsigned int nx, unsigned int ny, double[::1] extent):
    cdef np.ndarray back_projection = np.zeros([ny, nx], dtype=np.double)
    cdef double[:, ::1] back_projection_view = back_projection

    cdef double[::1] projection_view_1
    cdef double[:, ::1] projection_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef unsigned int i

    if isinstance(theta, float):
        projection_view_1 = projection
        theta_ = theta

        _back_project_parallel(theta_, &r[0], &projection_view_1[0], projection_view_1.shape[0],
            &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection

    elif isinstance(theta, np.ndarray):
        projection_view_2 = projection
        theta_view_1 = theta

        for i in range(theta_view_1.shape[0]):
            _back_project_parallel(theta_view_1[i], &r[0], &projection_view_2[i, 0], projection_view_2.shape[1],
                &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection


def back_project_fan(theta, double[::1] phi, double radius, np.ndarray projection, unsigned int nx, unsigned int ny, double[::1] extent):
    cdef np.ndarray back_projection = np.zeros([ny, nx], dtype=np.double)
    cdef double[:, ::1] back_projection_view = back_projection

    cdef double[::1] projection_view_1
    cdef double[:, ::1] projection_view_2

    cdef double theta_
    cdef double[::1] theta_view_1

    cdef unsigned int i

    if isinstance(theta, float):
        projection_view_1 = projection
        theta_ = theta

        _back_project_fan(theta_, &phi[0], radius, &projection_view_1[0], projection_view_1.shape[0],
            &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection

    elif isinstance(theta, np.ndarray):
        projection_view_2 = projection
        theta_view_1 = theta

        for i in range(theta_view_1.shape[0]):
            _back_project_fan(theta_view_1[i], &phi[0], radius, &projection_view_2[i, 0], projection_view_2.shape[1],
                &back_projection_view[0, 0], nx, ny, &extent[0])

        return back_projection
"""

def back_project_fan(double geometry_angle, double[::1] fan_angles, double radius,
                     double[::1] sinogram, double[:, ::1] backproject, double[::1] extent):

    _back_project_fan(geometry_angle, &fan_angles[0], fan_angles.shape[0], radius, &sinogram[0], 
                      &backproject[0, 0], backproject.shape[1], backproject.shape[0], &extent[0])

def s_back_project_fan(double[::1] geometry_angles, double[::1] fan_angles, double radius,
                       double[:, ::1] sinogram, unsigned int pixels_nx, unsigned int pixels_ny, double[::1] extent):
    cdef:
        np.ndarray backproject = np.zeros((pixels_ny, pixels_nx), dtype=np.double)
        double[:, ::1] backproject_view = backproject

    _s_back_project_fan(&geometry_angles[0], geometry_angles.shape[0], &fan_angles[0], fan_angles.shape[0], radius, &sinogram[0, 0], 
                        &backproject_view[0, 0], backproject.shape[1], backproject.shape[0], &extent[0])
    
    return backproject

def fission_probability(unsigned int k, double[::1] ray, double[:, ::1] mu, double[:, ::1] mu_f, double[:, ::1] p,
                        double[::1] extent, double[::1] detector_points, double[::1] nu, double step_size):
    cdef:
        unsigned int detector_points_n = <unsigned int>(detector_points.shape[0] / 4);

    return _fission_probability(k, &ray[0], &mu[0, 0], &mu_f[0, 0], &p[0, 0], &extent[0],
                                mu.shape[1], mu.shape[0], &detector_points[0], detector_points_n,
                                &nu[0], nu.shape[0], step_size)

def fission_forward_project_parallel(double geometry_angle, double[::1] parallel_coord, double length,
                                     unsigned int k, double[:, ::1] mu, double[:, ::1] mu_f, double[:, ::1] p,
                                     double[::1] extent, double[::1] detector_points, double[::1] nu, double step_size):

    cdef:
        np.ndarray fission_sinogram = np.zeros((parallel_coord.shape[0]), dtype=np.double)
        double[::1] fission_sinogram_view = fission_sinogram
        unsigned int detector_points_n = <unsigned int>(detector_points.shape[0] / 4);

    _fission_forward_project_parallel(
        geometry_angle, 
        &parallel_coord[0], parallel_coord.shape[0], 
        length, k, 
        &fission_sinogram_view[0], &mu[0, 0], &mu_f[0, 0], &p[0, 0], 
        &extent[0], mu.shape[1], mu.shape[0], 
        &detector_points[0], detector_points_n,
        &nu[0], nu.shape[0], 
        step_size)
    
    return fission_sinogram

def s_fission_forward_project_parallel(double[::1] geometry_angles, double[::1] parallel_coord, double length,
                                       unsigned int k, double[:, ::1] mu, double[:, ::1] mu_f, double[:, ::1] p,
                                       double[::1] extent, double[::1] detector_points, double[::1] nu, double step_size):

    cdef:
        np.ndarray fission_sinogram = np.zeros((geometry_angles.shape[0], parallel_coord.shape[0]), dtype=np.double)
        double[::1] fission_sinogram_view = fission_sinogram
        unsigned int detector_points_n = <unsigned int>(detector_points.shape[0] / 4);


    _s_fission_forward_project_parallel(
        &geometry_angles[0], geometry_angles.shape[0], 
        &parallel_coord[0], parallel_coord.shape[0], 
        length, k, 
        &fission_sinogram_view[0], &mu[0, 0], &mu_f[0, 0], &p[0, 0], 
        &extent[0], mu.shape[1], mu.shape[0], 
        &detector_points[0], detector_points_n,
        &nu[0], nu.shape[0], 
        step_size)
    
    return fission_sinogram

def precalculate_detector_probability(
    double[:, ::1] mu, double[::1] extent, double[::1] detector_points, double step_size):

    cdef:
        np.ndarray detector_prob = np.empty_like(mu)
        double[:, ::1] detector_prob_view = detector_prob
        unsigned int detector_points_n = <unsigned int>(detector_points.shape[0] / 4);
    
    _precalculate_detector_probability(
        &detector_prob_view[0, 0], &mu[0, 0],
        mu.shape[1], mu.shape[0], &extent[0], &detector_points[0], detector_points_n,
        step_size)
    
    return detector_prob

def fission_precalc_forward_project_parallel(
    double geometry_angle, double[::1] parallel_coord, double length, unsigned int k, 
    double[:, ::1] mu, double[:, ::1] mu_f, double[:, ::1] p, double[:, ::1] detector_prob,
    double[::1] extent, double[::1] nu, double step_size):

    cdef:
        np.ndarray fission_sinogram = np.zeros((parallel_coord.shape[0]), dtype=np.double)
        double[::1] fission_sinogram_view = fission_sinogram

    _fission_precalc_forward_project_parallel(
        geometry_angle, 
        &parallel_coord[0], parallel_coord.shape[0], 
        length, k, 
        &fission_sinogram_view[0], &mu[0, 0], &mu_f[0, 0], &p[0, 0], &detector_prob[0, 0],
        &extent[0], mu.shape[1], mu.shape[0], 
        &nu[0], nu.shape[0], 
        step_size)
    
    return fission_sinogram
"""