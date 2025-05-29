// hello.c
#include <Python.h>

// 这是一个简单的加法函数
static PyObject* add(PyObject* self, PyObject* args) {
    int a, b;
    
    // 解析传入的两个参数
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    // 返回加法结果
    return PyLong_FromLong(a + b);
}

// 这是一个简单的 hello 函数
static PyObject* hello(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", "Hello, world!");
}

// 方法表，列出所有要暴露给 Python 的 C 函数
static PyMethodDef methods[] = {
    {"add", add, METH_VARARGS, "Add two numbers"},
    {"hello", hello, METH_NOARGS, "Print 'Hello, world!'"},
    {NULL, NULL, 0, NULL}  // 结束符
};

// 模块定义
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "hello_module",  // 模块名称
    "A simple module written in C",
    -1,
    methods
};

// 模块初始化函数
PyMODINIT_FUNC PyInit_hello_module(void) {
    return PyModule_Create(&module);
}
