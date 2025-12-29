from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="traceable_autograd_demo",
    ext_modules=[
        CppExtension(
            name="traceable_autograd_demo",
            sources=["csrc/scale_mul.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
