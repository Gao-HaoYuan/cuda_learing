import os
from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext

import pybind11

CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

class BuildExt(build_ext):
  def run(self):
    os.system('bash ./build.sh')
    super().run()

ext = Extension(
    name='cuda_add',
    sources=['add.cpp'],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(CUDA_HOME, 'include')
    ],
    library_dirs=['./lib', os.path.join(CUDA_HOME, 'lib64')],
    libraries=['my_add', 'cudart'],
    extra_compile_args=['-Wno-sign-compare', '-O3'],
    language='c++'
)

setup(
    name='cuda_add',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExt},
)
