#include <Python.h>


typedef struct {
    PyObject_HEAD
    double *data;
    Py_ssize_t size;
} VectorObject;

static PyObject* Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    VectorObject *self;

    self = (VectorObject *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->data = NULL;
    self->size = 0;

    return (PyObject *)self;
}
static int Vector_init(VectorObject *self, PyObject *args, PyObject *kwds)
{
    Py_ssize_t size;

    if (!PyArg_ParseTuple(args, "n", &size))
        return -1;

    self->size = size;
    self->data = malloc(size * sizeof(double));

    if (!self->data)
        return -1;

    return 0;
}
static void Vector_dealloc(VectorObject *self)
{
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}




static int Vector_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    VectorObject *self = (VectorObject *)obj;

    return PyBuffer_FillInfo(
        view,
        obj,
        (void *)self->data,
        self->size * sizeof(double),
        0,
        flags
    );
}

static void Vector_releasebuffer(PyObject *obj, Py_buffer *view)
{

}

static PyBufferProcs Vector_bufferprocs = {
    .bf_getbuffer = Vector_getbuffer,
    .bf_releasebuffer = Vector_releasebuffer,
};

static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vector_buffer.Vector",
    .tp_basicsize = sizeof(VectorObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = (newfunc)Vector_new,
    .tp_init = (initproc)Vector_init,
    .tp_dealloc = (destructor)Vector_dealloc,
    .tp_as_buffer = &Vector_bufferprocs,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "vector_buffer",
    NULL,
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit_vector_buffer(void)
{
    PyObject *m;

    if (PyType_Ready(&VectorType) < 0)
        return NULL;

    m = PyModule_Create(&moduledef);
    if (!m)
        return NULL;

    Py_INCREF(&VectorType);
    PyModule_AddObject(m, "Vector", (PyObject *)&VectorType);

    return m;
}




