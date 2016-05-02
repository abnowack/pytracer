import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef np.ndarray[np.float64_t, ndim=2] calc_intersections(np.ndarray[np.float64_t, ndim=1] start,
                                                         np.ndarray[np.float64_t, ndim=1] end,
                                                         np.ndarray[np.float64_t, ndim=3] segments):
    cdef np.ndarray intercepts = np.zeros([20, 3], dtype=np.float64)
    cdef int count = 0

    cdef double epsilon = 1e-15
    cdef u, t, u_num, t_num, denom

    cdef int segments_n = segments.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] p, q, r, s

    for seg_i in range(segments_n):
        # calc intercept
        p = segments[seg_i][0]
        q = start
        r = segments[seg_i][1] - segments[seg_i][0]
        s = end - start

        denom = r[0] * s[1] - r[1] * s[0]

        # colinear or parallel
        if denom == 0.:
            continue

        u_num = (q - p)[0] * r[1] - (q - p)[1] * r[0]
        t_num = (q - p)[0] * s[1] - (q - p)[1] * s[0]
        t = t_num / denom
        u = u_num / denom

        if (-epsilon < t < 1 + epsilon) and (0. < u <= 1.):
            intercepts[count, 0] = p[0] + t * r[0]
            intercepts[count, 1] = p[1] + t * r[1]
            intercepts[count, 2] = seg_i
            count += 1
        else:
            continue

    intercepts = intercepts[:count-1]
    return intercepts
