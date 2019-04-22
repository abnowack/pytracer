cimport numpy as np

np.import_array()

cdef extern from "doubleit.h":
    void doubleit (double * in_array, double * out_array, int size)

def doubleit_func(double[::1] in_array, double[::1] out_array):
    doubleit(<double*> np.PyArray_DATA(in_array), <double*> np.PyArray_DATA(out_array),
             in_array.shape[0])