import os

from glob import glob
from platform import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

class BuildHaste(BuildExtension):
  def run(self):
    # os.system('make my_add LIBNAME=libmy_add.a')
    os.system('bash ./build.sh')
    super().run()

base_path = os.path.dirname(os.path.realpath(__file__))
if 'Windows' in platform():
  CUDA_HOME = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
  extra_args = []
else:
  CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
  extra_args = ['-Wno-sign-compare']

have_make = CUDAExtension(
                    name = 'my_add',
                    sources = glob('./my_add/*.cpp'),
                    include_dirs=[os.path.join(base_path, './')],
                    extra_compile_args = extra_args,
                    libraries = ['my_add'],
                    library_dirs = ['./lib']
                  )

setup(
    name ="my_add",
    ext_modules = [have_make],
    cmdclass = {"build_ext": BuildHaste}
)
