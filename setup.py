import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('mdstates', 'version.py')
with open(ver_file) as f:
    exec(f.read())


ext_modules = [Extension(
               name="mdstates.hmm_cython",
               sources=["mdstates/hmm_cython.pyx", "mdstates/hmm.cpp"],
               # include_dirs=["/Users/ldgibson/Development/mdstates/mdstates/.", np.get_include()],
               extra_compile_args=["-std=c++11"],
               language="c++",
               )]

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES,
            # cmdclass={'build_ext': build_ext},
            ext_modules=cythonize(ext_modules),
            # requires=REQUIRES
            )


if __name__ == '__main__':
    setup(**opts)
