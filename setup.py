from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import sys

system = platform.system()

extra_compile_args = []
extra_link_args = []

if system == "Linux":
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif system == "Darwin":
    extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
elif system == "Windows":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []

extensions = [
    Extension(
        name="bh_potential",
        sources=["bh_potential.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="bh_potential",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)