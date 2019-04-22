import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "tomo.h":
    void _line_box_crop(double * box, double * line, double * crop_line);
    void _s_line_box_crop(double * box, double * lines, double * crop_lines, int line_size);
    double _bilinear_interpolation(double x, double y,
                               double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                               double * extent);
    double _s_bilinear_interpolation(double * x, double * y, double * values,
                                 double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                 double * extent)

def ray_box_crop(double[::1] ray, double[::1] extent):
    cdef:
        np.ndarray crop_ray = np.empty_like(ray)
        double[::1] crop_ray_view = crop_ray

    _line_box_crop(&extent[0], &ray[0], &crop_ray_view[0])
    return crop_ray

def s_ray_box_crop(double[:, ::1] rays, double[::1] extent):
    cdef:
        np.ndarray crop_rays = np.empty_like(rays)
        double[:, ::1] crop_rays_view = crop_rays

    _s_line_box_crop(&extent[0], &rays[0, 0], &crop_rays_view[0, 0], rays.shape[0])
    return crop_rays

def bilinear_interpolation(double x, double y, double[:, ::1] pixels, double[::1] extent):
    cdef:
        double value

    value = _bilinear_interpolation(x, y, &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])
    return value

def s_bilinear_interpolation(double[::1] x, double[::1] y, double[:, ::1] pixels, double[::1] extent):
    cdef:
        np.ndarray values = np.empty_like(x)
        double[::1] values_view = values

    _s_bilinear_interpolation(&x[0], &y[0], &values_view[0], &pixels[0, 0], pixels.shape[1], pixels.shape[0], &extent[0])
    return values