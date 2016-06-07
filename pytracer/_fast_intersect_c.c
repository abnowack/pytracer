#include <Python.h>
#include <numpy/arrayobject.h>

union Point {
    double p[2];
    struct {
        double x;
        double y;
    };
};

int intersect(union Point s1, union Point s2, union Point o1, union Point o2, union Point *intersect) {
    double epsilon;
    double r_x, r_y, s_x, s_y;
    double denom, u_num, t_num, t, u;

    epsilon = 1e-15;

    r_x = s2.x - s1.x;
    r_y = s2.y - s1.y;
    s_x = o2.x - o1.x;
    s_y = o2.y - o1.y;

    denom = r_x * s_y - r_y * s_x;

    if (denom == 0.0) {
        return -1;
    }

    u_num = (o1.x - s1.x) * r_y - (o1.y - s1.y) * r_x;
    t_num = (o1.x - s1.x) * s_y - (o1.y - s1.y) * s_x;

    t = t_num / denom;
    u = u_num / denom;

    intersect->x = s1.x + t * r_x;
    intersect->y = s1.y + t * r_y;

    if ((-epsilon < t) && (t < 1.0 - epsilon)) {
        return 0;
    }

    return -1;
}

static PyObject* intersect_c(PyObject* self, PyObject *args) {
    union Point s1, s2, o1, o2, isect;
    int ret;

    if (!PyArg_ParseTuple(args, "dddddddd", &s1.x, &s1.y, &s2.x, &s2.y, &o1.x, &o1.y, &o2.x, &o2.y))
        return NULL;

    ret = intersect(s1, s2, o1, o2, &isect);

    if (ret != 0)
        Py_RETURN_NONE;

    return Py_BuildValue("ff", isect.x, isect.y);
}

static PyObject* intersecting_segments_c(PyObject* self, PyObject *args) {
    PyArrayObject *segs;
    int segs_ndims;
    npy_intp *segs_dims;

    double *x1, *x2, *y1, *y2;
    union Point s1, s2, o1, o2, isect;
    int ret;

    PyObject *seg_index;
    PyObject *xcoord_float;
    PyObject *ycoord_float;

    PyObject *index_list;
    PyObject *xcoord_list;
    PyObject *ycoord_list;
    PyObject *return_list;

    long int i;

    index_list = PyList_New(0);
    xcoord_list = PyList_New(0);
    ycoord_list = PyList_New(0);
    return_list = PyList_New(0);

    if (!PyArg_ParseTuple(args, "O!dddd", &PyArray_Type, &segs, &o1.x, &o1.y, &o2.x, &o2.y))
        return NULL;

    // Get dimension of segs
    segs_ndims = PyArray_NDIM(segs);
    segs_dims = PyArray_DIMS(segs);

    for (i = 0; i < segs_dims[0]; i++) {
        x1 = (double*)PyArray_GETPTR3(segs, i, 0, 0);
        x2 = (double*)PyArray_GETPTR3(segs, i, 1, 0);
        y1 = (double*)PyArray_GETPTR3(segs, i, 0, 1);
        y2 = (double*)PyArray_GETPTR3(segs, i, 1, 1);
        s1.x = *x1;
        s1.y = *y1;
        s2.x = *x2;
        s2.y = *y2;

        ret = intersect(s1, s2, o1, o2, &isect);

        if (ret == 0) {
            seg_index = PyInt_FromLong(i);
            xcoord_float = PyFloat_FromDouble(isect.x);
            ycoord_float = PyFloat_FromDouble(isect.y);
            PyList_Append(index_list, seg_index);
            PyList_Append(xcoord_list, xcoord_float);
            PyList_Append(ycoord_list, ycoord_float);
        }
    }

    // Cleanup created vars
//    Py_DECREF(segs);

    PyList_Append(return_list, index_list);
    PyList_Append(return_list, xcoord_list);
    PyList_Append(return_list, ycoord_list);
    return return_list;

  fail:
    Py_XDECREF(segs);
    return NULL;
}

static PyMethodDef IntersectMethods[] =
{
    {"intersect_c", intersect_c, METH_VARARGS, "intersect in C"},
    {"intersecting_segments_c", intersecting_segments_c, METH_VARARGS, "intersecting segments in C"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC

initintersect_module(void)
{
    (void) Py_InitModule("intersect_module", IntersectMethods);
    import_array();
}