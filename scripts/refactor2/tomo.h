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

double _detect_probability(
    double point[2],
    double *mu, unsigned int mu_nx, unsigned int mu_ny, double extent[4],
    double detector_points[][4], unsigned int n, double step_size);

double _fission_forward_project(
    double ray[4], unsigned int k,
    double *mu_pixels, double *mu_f_pixels, double *p_pixels, double *detect_prob,
    double extent[4], unsigned int nx, unsigned int ny,
    double nu_dist[], unsigned int nu_dist_n, double step_size);