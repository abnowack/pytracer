import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "tomo.h":
    void _ray_box_crop(double *extent, double *ray, double *crop_ray);
    void _s_ray_box_crop(double *extent, double *rays, double *crop_rays, unsigned int n_rays);
    double _bilinear_interpolate(
        double x, double y,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    void _s_bilinear_interpolate(
        double *x, double *y, double *z, unsigned int x_n,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    double _raytrace_bilinear(
        double *ray, double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    double _s_raytrace_bilinear(
        double *rays, unsigned int rays_n, double *values,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    void _fan_ray(double *ray, double x1, double x2, double radius);
    void _s_fan_ray(double *rays, double x1, double *x2, unsigned int x2_n, double radius);
    void _parallel_ray(double *ray, double x1, double x2, double length);
    void _s_parallel_ray(double *rays, double x1, double *x2, unsigned int x2_n, double length);
    void _forward_project_fan(
        double x1, double *x2, unsigned int x2_n, double radius, double *values,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    void _s_forward_project_fan(
        double *x1, unsigned int x1_n, double *x2, unsigned int x2_n,
        double radius, double *values,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    void _back_project_fan(
        double x1, double *x2, unsigned int x2_n, double radius, double *sinogram,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    void _s_back_project_fan(
        double *x1, unsigned int x1_n, double *x2, unsigned int x2_n, double radius, double *sinogram,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    void _forward_project_parallel(
        double x1, double *x2, unsigned int x2_n, double length, double *values,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    void _s_forward_project_parallel(
        double *x1, unsigned int x1_n, double *x2, unsigned int x2_n,
        double length, double *values,
        double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double step_size);
    double _gsl_test(double x);
    void _parallel_detector(
        double *detector_points, unsigned int detectors_n,
        double x1, double width, double radius);
    void _fan_detector(
        double *detector_points, unsigned int detectors_n,
        double x1, double detector_angle, double radius);
    void _back_project_parallel(
        double x1, double *x2, unsigned int x2_n, double *sinogram,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    void _s_back_project_parallel(
        double *x1, unsigned int x1_n, double *x2, unsigned int x2_n, double *sinogram,
        double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent);
    double _detect_probability(
        double *point, double *pixels, unsigned int pixels_nx, unsigned int pixels_ny,
        double *extent, double *detector_points, unsigned int detector_points_n, double step_size);
    double _fission_probability(
        unsigned int k, double *ray,
        double *mu_pixels, double *mu_f_pixels, double *p_pixels,
        double *extent, unsigned int pixels_nx, unsigned int pixels_ny,
        double *detector_points, unsigned int detector_points_n,
        double *nu_dist, unsigned int nu_dist_n, double step_size);
    void _fission_forward_project_parallel(
        double x1, double *x2, unsigned int x2_n, double length,
        unsigned int k, double *values,
        double *mu_pixels, double *mu_f_pixels, double *p_pixels,
        double *extent, unsigned int pixels_nx, unsigned int pixels_ny,
        double *detector_points, unsigned int detector_points_n,
        double *nu_dist, unsigned int nu_dist_n, double step_size);
    void _s_fission_forward_project_parallel(
        double *x1, unsigned int x1_n, double *x2, unsigned int x2_n, double length,
        unsigned int k, double *values,
        double *mu_pixels, double *mu_f_pixels, double *p_pixels,
        double *extent, unsigned int pixels_nx, unsigned int pixels_ny,
        double *detector_points, unsigned int detector_points_n,
        double *nu_dist, unsigned int nu_dist_n, double step_size);
    void _precalculate_detector_probability(
        double *values, double *mu,
        unsigned int pixels_nx, unsigned int pixels_ny, double *extent,
        double *detector_points, unsigned int detector_points_n, double step_size);
    void _fission_precalc_forward_project_parallel(
        double x1, double *x2, unsigned int x2_n, double length,
        unsigned int k, double *values,
        double *mu_pixels, double *mu_f_pixels, double *p_pixels, double *detect_prob,
        double *extent, unsigned int pixels_nx, unsigned int pixels_ny,
        double *nu_dist, unsigned int nu_dist_n, double step_size);


def ray_box_crop(double[::1] extent not None, double[::1] ray not None):
    cdef:
        np.ndarray crop_ray = np.empty_like(ray)
        double[::1] crop_ray_view = crop_ray

    _ray_box_crop(&extent[0], &ray[0], &crop_ray_view[0])
    return crop_ray

def s_ray_box_crop(double[::1] extent not None, double[:, ::1] rays not None):
    cdef:
        np.ndarray crop_rays = np.empty_like(rays)
        double[:, ::1] crop_rays_view = crop_rays

    _s_ray_box_crop(&extent[0], &rays[0, 0], &crop_rays_view[0, 0], rays.shape[0])
    return crop_rays

def bilinear_interpolate(double x, double y,
                         double[:, ::1] pixels not None, double[::1] extent not None):
    
    return _bilinear_interpolate(x, y, &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])

def s_bilinear_interpolate(double[::1] xs, double[::1] ys,
                           double[:, ::1] pixels not None, double[::1] extent not None):
    cdef:
        np.ndarray zs = np.empty_like(xs)
        double[::1] zs_view = zs
    
    _s_bilinear_interpolate(&xs[0], &ys[0], &zs_view[0], xs.shape[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])
    return zs

def raytrace_bilinear(double[::1] ray not None, double[:, ::1] pixels not None, 
                      double[::1] extent not None, double step_size):
    return _raytrace_bilinear(&ray[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0], step_size)

def s_raytrace_bilinear(double[:, ::1] rays not None, double[:, ::1] pixels not None, 
                        double[::1] extent not None, double step_size):
    cdef:
        np.ndarray values = np.zeros(rays.shape[0], dtype=np.double)
        double[::1] values_view = values

    _s_raytrace_bilinear(&rays[0, 0], rays.shape[0], &values_view[0],
                         &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0], step_size)
    
    return values

def fan_ray(double geometry_angle, double fan_angle, double radius):
    cdef:
        np.ndarray ray = np.zeros(4, dtype=np.double)
        double[::1] ray_view = ray
    
    _fan_ray(&ray_view[0], geometry_angle, fan_angle, radius)
    return ray

def s_fan_ray(double geometry_angle, double[::1] fan_angles, double radius):
    cdef:
        np.ndarray rays = np.zeros((fan_angles.shape[0], 4), dtype=np.double)
        double[:, ::1] rays_view = rays

    _s_fan_ray(&rays_view[0, 0], geometry_angle, &fan_angles[0], fan_angles.shape[0], radius)
    return rays

def parallel_ray(double geometry_angle, double parallel_coord, double length):
    cdef:
        np.ndarray ray = np.zeros(4, dtype=np.double)
        double[::1] ray_view = ray
    
    _parallel_ray(&ray_view[0], geometry_angle, parallel_coord, length)
    return ray

def s_parallel_ray(double geometry_angle, double[::1] parallel_coords, double length):
    cdef:
        np.ndarray rays = np.zeros((parallel_coords.shape[0], 4), dtype=np.double)
        double[:, ::1] rays_view = rays

    _s_parallel_ray(&rays_view[0, 0], geometry_angle, 
                              &parallel_coords[0], parallel_coords.shape[0], length)
    return rays

def forward_project_fan(double geometry_angle, double[::1] fan_angles, double radius,
                        double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros(fan_angles.shape[0], dtype=np.double)
        double[::1] values_view = values
    
    _forward_project_fan(geometry_angle, &fan_angles[0], fan_angles.shape[0], radius,
                         &values_view[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                         &extent[0], step_size)
    return values

def s_forward_project_fan(double[::1] geometry_angles, double[::1] fan_angles, double radius,
                          double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros((geometry_angles.shape[0], fan_angles.shape[0]), dtype=np.double)
        double[:, ::1] values_view = values
    
    _s_forward_project_fan(&geometry_angles[0], geometry_angles.shape[0], 
                           &fan_angles[0], fan_angles.shape[0], radius,
                           &values_view[0, 0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                           &extent[0], step_size)
    return values

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

def forward_project_parallel(double geometry_angle, double[::1] parallel_coord, double length,
                             double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros(parallel_coord.shape[0], dtype=np.double)
        double[::1] values_view = values
    
    _forward_project_parallel(geometry_angle, &parallel_coord[0], parallel_coord.shape[0], length,
                              &values_view[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                              &extent[0], step_size)
    return values

def s_forward_project_parallel(double[::1] geometry_angles, double[::1] parallel_coord, double length,
                               double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros((geometry_angles.shape[0], parallel_coord.shape[0]), dtype=np.double)
        double[:, ::1] values_view = values
    
    _s_forward_project_parallel(&geometry_angles[0], geometry_angles.shape[0], 
                                &parallel_coord[0], parallel_coord.shape[0], length,
                                &values_view[0, 0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                                &extent[0], step_size)
    return values

def gsl_test(double x):
    return _gsl_test(x)

def parallel_detector(double geometry_angle, unsigned int n_detectors, double width, double radius):
    cdef:
        np.ndarray detector_points = np.empty(4 * n_detectors, dtype=np.double)
        double[::1] detector_points_view = detector_points
    
    _parallel_detector(&detector_points_view[0], n_detectors, geometry_angle, width, radius)

    return detector_points

def fan_detector(double geometry_angle, unsigned int n_detectors, double detector_angle, double radius):
    cdef:
        np.ndarray detector_points = np.empty(4 * n_detectors, dtype=np.double)
        double[::1] detector_points_view = detector_points
    
    _fan_detector(&detector_points_view[0], n_detectors, geometry_angle, detector_angle, radius)

    return detector_points

def back_project_parallel(double geometry_angle, double[::1] parallel_coord,
                          double[::1] sinogram, double[:, ::1] backproject, double[::1] extent):
    
    _back_project_parallel(geometry_angle, &parallel_coord[0], parallel_coord.shape[0],
                           &sinogram[0], &backproject[0, 0], backproject.shape[1], backproject.shape[0], &extent[0])

def s_back_project_parallel(double[::1] geometry_angles, double[::1] parallel_coord,
                           double[:, ::1] sinogram, unsigned int pixels_nx, unsigned int pixels_ny, double[::1] extent):
    cdef:
        np.ndarray backproject = np.zeros((pixels_ny, pixels_nx), dtype=np.double)
        double[:, ::1] backproject_view = backproject

    _s_back_project_parallel(&geometry_angles[0], geometry_angles.shape[0], &parallel_coord[0], parallel_coord.shape[0], &sinogram[0, 0], 
                             &backproject_view[0, 0], pixels_nx, pixels_ny, &extent[0])
    
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