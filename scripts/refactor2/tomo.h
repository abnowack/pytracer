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