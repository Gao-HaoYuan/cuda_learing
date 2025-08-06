import os

import torch
from glob import glob
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ENABLE_DEBUG = 'OFF'

class BuildCuKernel(BuildExtension):
  def run(self):
      torch_toolchain = torch.utils.cmake_prefix_path

      cmd = 'bash ./build.sh'
      cmd += f' --torch-toolchain {torch_toolchain}'
      cmd += f' --enable-debug {ENABLE_DEBUG}'

      os.system(cmd)
      super().run()

base_path = os.path.dirname(os.path.realpath(__file__))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
setup(
    name ="cuKernel",
    ext_modules=[
        CUDAExtension(
            name = 'cuKernel',
            sources = glob(os.path.join(base_path, 'witin_nn', 'cu', 'register', '*.cpp')),
            libraries = ['cuKernel'],
            library_dirs = ['./lib'],
            include_dirs = [
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(base_path, 'witin_nn', 'cu', 'include'),
            ]
        )
    ],
    cmdclass = {"build_ext": BuildCuKernel}
)
