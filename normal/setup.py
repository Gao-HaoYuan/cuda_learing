# setup.py
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="AFP",
    ext_modules=[
        CUDAExtension(
            name="AFP",
            sources=["binding.cpp", "kernel.cu"],
            include_dirs=torch.utils.cpp_extension.include_paths(),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
