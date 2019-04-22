void _line_box_crop(double * box, double line[4], double crop_line[4]);
void _s_line_box_crop(double * box, double lines[][4], double crop_lines[][4], int line_size);
double _bilinear_interpolation(double x, double y,
                               double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                               double * extent);
double _s_bilinear_interpolation(double * x, double * y, double * values,
                                 double * pixels, unsigned int pixels_nx, unsigned int pixels_ny,
                                 double * extent)
