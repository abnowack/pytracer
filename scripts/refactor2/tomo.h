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
/*
void _back_project_fan(
    double x1, double *x2, unsigned int x2_n, double radius, double *sinogram,
    double *backproject, unsigned int pixels_nx, unsigned int pixels_ny,
    double *extent);

double _detect_probability(
    double pos[2],
    double *mu, unsigned int mu_nx, unsigned int mu_ny,
    double detector_points[][4], unsigned int detector_points_n,
    double extent[4], double step_size);

void _precalculate_detector_probability(
    double *values,
    double *mu, unsigned int nx, unsigned int ny,
    double *detector_points, unsigned int detector_points_n,
    double extent[4], double step_size);

double _fission_probability(
    double ray[4],
    double *mu, double *mu_f, double *p, double *detect_prob,
    unsigned int nx, unsigned int ny,
    unsigned int k, double nu_dist[], unsigned int nu_dist_n,
    double extent[4], double step_size);

void _fission_forward_project(
    unsigned int n,
    double values[], double rays[][4],
    double *mu, double *mu_f, double *p, double *detect_prob,
    unsigned int nx, unsigned int ny,
    unsigned int k, double nu_dist[], unsigned int nu_dist_n,
    double extent[4], double step_size);
    */