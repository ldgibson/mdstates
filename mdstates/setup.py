from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="hmm_cython",
    sources=["hmm_cython.pyx", "hmm.cpp"],
    extra_compile_args=["-std=c++11"],
    language="c++",
    )]

setup(
    name='hmm_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    )
