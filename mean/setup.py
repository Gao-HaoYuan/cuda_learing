# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mean_hd',
    ext_modules=[
        CUDAExtension(
            name='mean_hd',  # import mean_hd
            sources=['binding.cpp', 'kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
