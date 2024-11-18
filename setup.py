from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

openmp_flags = []
if os.name == 'posix':  # On Linux/macOS
    openmp_flags = ['-fopenmp']
elif os.name == 'nt':  # On Windows
    openmp_flags = ['/openmp']

extensions = [
    Extension(
        name="bh_potential",
        sources=["bh_potential.pyx"],
        include_dirs=[np.get_include()],  # Include directories
        extra_compile_args=openmp_flags,  # OpenMP compile flags
        extra_link_args=openmp_flags,  # OpenMP link flags
    )
]

setup(
    ext_modules=cythonize(extensions),
)
