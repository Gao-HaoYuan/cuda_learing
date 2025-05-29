from setuptools import setup, Extension

module = Extension(
    name = "hello",
    sources = ['add.c']
)

setup(
    name="hello_test",
    version="0.1.0",
    ext_modules=[module],
    python_requires=">=3.6"
)
