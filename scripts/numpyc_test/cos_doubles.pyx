""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

import numpy as np
# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "cos_doubles.h":
    void cos_doubles (double * in_array, double * out_array, int size)
    void cos_doubles2d(double * in_array, double * out_array, int size1, int size2)

# create the wrapper code, with numpy type annotations
def cos_doubles_func(np.ndarray[double, ndim=1, mode="c"] in_array not None,
                     np.ndarray[double, ndim=1, mode="c"] out_array not None):
    cos_doubles(<double*> np.PyArray_DATA(in_array),
                <double*> np.PyArray_DATA(out_array),
                in_array.shape[0])

def cos_doubles_func2(double[::1] in_array not None,
                      double[::1] out_array not None):
    cos_doubles(&in_array[0],
                &out_array[0],
                in_array.shape[0])


# def cos_doubles_func3(double[::1] in_array not None):
#     cdef:
#         np.ndarray out_array = np.empty_like(in_array)
#         double[::1] out_array_view = out_array
#
#     cos_doubles(&in_array[0], &out_array_view[0], in_array.shape[0])
#
#     return out_array


def cos_doubles_func2D(double[:, ::1] in_array not None):
    cdef:
        np.ndarray out_array = np.empty_like(in_array)
        double[:, ::1] out_array_view = out_array

    cos_doubles2d(&in_array[0, 0], &out_array_view[0, 0], in_array.shape[0], in_array.shape[1])

    return out_array
