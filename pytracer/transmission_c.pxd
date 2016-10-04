cpdef point_segment_distance(double px, double py, double x0, double x1, double y0, double y1)

cpdef absorbance_at_point(double point_x, double point_y, double[:, :, ::1] segments,
                          double[:, ::1] absorbance)

cpdef absorbance_image(double[:, ::1] image, double[::1] xs, double[::1] ys,
                       double[:, :, ::1] segments, double[:, ::1] absorbance)

cpdef int intersections(double[::1] start, double[::1] end, double[:, :, ::1] segments,
                        double[:, ::1] intersect_cache, int[::1] index_cache, bint ray)

cpdef double absorbance(double[::1] start, double[::1] end,
                        double[:, :, ::1] segments, double[:, ::1] seg_absorption,
                        double universe_absorption, double[:, ::1] intersect_cache,
                        int[::1] index_cache)

cpdef void absorbances(double[:, ::1] start, double[:, ::1] end,
                       double[:, :, ::1] segments, double[:, ::1] seg_absorption,
                       double universe_absorption, double[:, ::1] intersects_cache,
                       int[::1] indexes_cache, double[:] absorbance_cache)
