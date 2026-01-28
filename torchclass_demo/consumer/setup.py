from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os, torch

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

setup(
    name="consumer",
    ext_modules=[
        CppExtension(
            name="consumer",
            sources=["consumer.cpp"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
            extra_link_args=[f"-Wl,-rpath,{torch_lib}"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
