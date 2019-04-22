cimport numpy as np

np.import_array()

cdef extern from "doubleit.h":
    void doubleit (double * in_array, double * out_array, int size)

def doubleit_func(np.ndarray[double, ndim=1, mode="c"] in_array, np.ndarray[double, ndim=1, mode="c"] out_array):
    doubleit(<double*> np.PyArray_DATA(in_array), <double*> np.PyArray_DATA(out_array),
             in_array.shape[0])