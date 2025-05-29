# setup.py
from setuptools import setup, Extension

# 创建 C 扩展
module = Extension('hello_module', sources=['hello.c'])

# 调用 setup 函数，定义包的信息
setup(
    name='hello_project',
    version='0.1.0',
    ext_modules=[module],
)
