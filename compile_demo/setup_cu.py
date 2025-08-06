import os

from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ENABLE_DEBUG = False

# get required path.
base_path = os.path.dirname(os.path.realpath(__file__))

# ENABLE_DEBUG: enable debug log.
extra_args_cxx = ['-std=c++17', '-O3', '-Wno-sign-compare', '-fPIC']
extra_args_nvcc = ['-std=c++17', '-O3', '--extended-lambda', '--expt-relaxed-constexpr']

if ENABLE_DEBUG:
  extra_args_cxx.append('-DWITIN_DEBUG')
  extra_args_nvcc.append('-DWITIN_DEBUG')

extra_args_nvcc += [
    '-Xcompiler', '-Wno-sign-compare',
    '-Xcompiler', '-fPIC',
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_89,code=sm_89',
    '-gencode=arch=compute_90,code=sm_90'
]

sources = []
sources += glob(os.path.join(base_path, 'witin_nn', 'cu', 'src', '*.cu'))
sources += glob(os.path.join(base_path, 'witin_nn', 'cu', 'src', '*.cpp'))

CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
extension = CUDAExtension(
              name = 'cuKernel',
              sources = sources,
              include_dirs = [
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(base_path, 'witin_nn', 'cu', 'include')
              ],
              extra_compile_args = {
                  'cxx': extra_args_cxx,
                  'nvcc': extra_args_nvcc
              }
            )

setup(
    name ="cuKernel",
    ext_modules = [extension],
    cmdclass = {"build_ext": BuildExtension}
)
