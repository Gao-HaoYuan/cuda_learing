from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

setup(
    name="producer",
    ext_modules=[
        CppExtension(
            name="producer",                 # 生成 producer*.so
            sources=["producer.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
