from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='Hello world app',
    ext_modules=cythonize("z3_1.pyx"),
    zip_safe=False,
    include_dirs=[np.get_include()]
)
