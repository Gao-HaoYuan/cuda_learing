from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='extension_cpp',
    ext_modules=[
        CUDAExtension(
            name='extension_cpp',
            sources=[
                'custom_linear/linear_cpu.cpp',
                'custom_linear/linear_cuda.cu',
                'custom_linear/register.cpp',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
