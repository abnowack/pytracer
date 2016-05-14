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
        return Py_None;

    return Py_BuildValue("ff", isect.x, isect.y);
}

static PyObject* intersecting_segments_c(PyObject* self, PyObject *args) {
    PyObject *arg1=NULL;
    PyObject *segs=NULL;

    int segs_ndims;
    npy_intp *segs_dims;

    PyObject *return_list;
    int i;

    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    segs = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (segs == NULL) return NULL;

    // Get dimension of segs
    segs_ndims = PyArray_NDIM(segs);
    segs_dims = PyArray_DIMS(segs);

    // Get Point1,2 from oseg
    // Loop over dimension of segs and calculate intercept
    // Append to list (TODO: Array?)

    // Return intersections and intercepts

    return_list = PyTuple_New(segs_ndims);
    for (i = 0; i < segs_ndims; i++) {
        PyObject *num = PyLong_FromLong(segs_dims[i]);
        PyTuple_SetItem(return_tuple, i, num);
    }

    // Cleanup created vars
    Py_DECREF(segs);

    return return_tuple;

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