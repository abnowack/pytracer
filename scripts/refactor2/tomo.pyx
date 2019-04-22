import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "tomo.h":
    void _ray_box_crop(double * extent, double * ray, double * crop_ray)
    void _s_ray_box_crop(double * extent, double * rays, double * crop_rays, unsigned int n_rays)
    double _bilinear_interpolate(double x, double y,
                                 double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                 double * extent)
    void _s_bilinear_interpolate(double * x, double * y, double * z, unsigned int x_n,
                                 double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                 double * extent)
    double _raytrace_bilinear(double * ray, double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                              double * extent, double step_size)
    double _s_raytrace_bilinear(double * rays, unsigned int rays_n, double * values,
                                double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                double * extent, double step_size)
    void _calculate_fan_ray(double * ray, double x1, double x2, double radius)
    void _s_calculate_fan_ray(double * rays, double x1, double * x2, unsigned int x2_n, double radius)
    void _calculate_parallel_ray(double * ray, double x1, double x2, double length)
    void _s_calculate_parallel_ray(double * rays, double x1, double * x2, unsigned int x2_n, double length)
    void _forward_project_fan(double x1, double * x2, unsigned int x2_n, double radius, double * values, 
                              double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                              double * extent, double step_size)
    void _s_forward_project_fan(double * x1, unsigned int x1_n, double * x2, unsigned int x2_n, 
                                double radius, double * values, 
                                double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                double * extent, double step_size)
    void _back_project_fan(double x1, double * x2, unsigned int x2_n, double radius, double * sinogram, 
                           double * backproject, unsigned int pixels_nx, unsigned int pixels_ny,
                           double * extent)
    void _s_back_project_fan(double * x1, unsigned int x1_n, double * x2, unsigned int x2_n, double radius, double * sinogram, 
                             double * backproject, unsigned int pixels_nx, unsigned int pixels_ny,
                             double * extent)
    void _forward_project_parallel(double x1, double * x2, unsigned int x2_n, double length, double * values,
                                   double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                   double * extent, double step_size)
    void _s_forward_project_parallel(double * x1, unsigned int x1_n, double * x2, unsigned int x2_n, 
                                    double length, double * values, 
                                    double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                    double * extent, double step_size)
    double _gsl_test(double x)
    void _back_project_parallel(double x1, double * x2, unsigned int x2_n, double * sinogram, 
                                double * backproject, unsigned int pixels_nx, unsigned int pixels_ny,
                                double * extent)
    void _s_back_project_parallel(double * x1, unsigned int x1_n, double * x2, unsigned int x2_n, double * sinogram, 
                                  double * backproject, unsigned int pixels_nx, unsigned int pixels_ny,
                                  double * extent)

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

def calculate_fan_ray(double geometry_angle, double fan_angle, double radius):
    cdef:
        np.ndarray ray = np.zeros(4, dtype=np.double)
        double[::1] ray_view = ray
    
    _calculate_fan_ray(&ray_view[0], geometry_angle, fan_angle, radius)
    return ray

def s_calculate_fan_ray(double geometry_angle, double[::1] fan_angles, double radius):
    cdef:
        np.ndarray rays = np.zeros((fan_angles.shape[0], 4), dtype=np.double)
        double[:, ::1] rays_view = rays

    _s_calculate_fan_ray(&rays_view[0, 0], geometry_angle, &fan_angles[0], fan_angles.shape[0], radius)
    return rays

def calculate_parallel_ray(double geometry_angle, double parallel_coord, double length):
    cdef:
        np.ndarray ray = np.zeros(4, dtype=np.double)
        double[::1] ray_view = ray
    
    _calculate_parallel_ray(&ray_view[0], geometry_angle, parallel_coord, length)
    return ray

def s_calculate_parallel_ray(double geometry_angle, double[::1] parallel_coords, double length):
    cdef:
        np.ndarray rays = np.zeros((parallel_coords.shape[0], 4), dtype=np.double)
        double[:, ::1] rays_view = rays

    _s_calculate_parallel_ray(&rays_view[0, 0], geometry_angle, 
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

def forward_project_parallel(double geometry_angle, double[::1] parallel_coord, double radius,
                             double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros(parallel_coord.shape[0], dtype=np.double)
        double[::1] values_view = values
    
    _forward_project_parallel(geometry_angle, &parallel_coord[0], parallel_coord.shape[0], radius,
                              &values_view[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                              &extent[0], step_size)
    return values

def s_forward_project_parallel(double[::1] geometry_angles, double[::1] parallel_coord, double radius,
                               double[:, ::1] pixels, double[::1] extent, double step_size):
    cdef:
        np.ndarray values = np.zeros((geometry_angles.shape[0], parallel_coord.shape[0]), dtype=np.double)
        double[:, ::1] values_view = values
    
    _s_forward_project_parallel(&geometry_angles[0], geometry_angles.shape[0], 
                                &parallel_coord[0], parallel_coord.shape[0], radius,
                                &values_view[0, 0], &pixels[0, 0], pixels.shape[1], pixels.shape[0],
                                &extent[0], step_size)
    return values

def gsl_test(double x):
    return _gsl_test(x)

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

