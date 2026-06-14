from mlx.extension import CMakeBuild, CMakeExtension
from setuptools import setup

setup(
    ext_modules=[CMakeExtension("sglang_mlx._ext")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"sglang_mlx": ["*.so", "*.dylib", "*.metallib"]},
    zip_safe=False,
)
