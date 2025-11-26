from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cublaslt_gemm',
    ext_modules=[
        CUDAExtension(
            name='cublaslt_gemm',
            sources=[
                'register.cpp',  # 上面的 pybind11 接口文件
                'witin_cublaslt.cpp'  # 你的 CublasLtGemmRunner 实现
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_80',  # 根据你的 GPU 修改架构，例如 sm_86、sm_90
                    '-std=c++17'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
