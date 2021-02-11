from os.path import join as pjoin, dirname, exists
from copy import deepcopy
import sys
import platform
import versioneer
import thetan.__info__ as info


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 4)
if sys.version_info < min_version:
    error = """
thetan does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*sys.version_info[:2], *min_version)
    sys.exit(error)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'):
    os.remove('MANIFEST')

# force_setuptools can be set from the setup_egg.py script
if 'force_setuptools' not in globals():
    # For some commands, always use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_mpkg', 'bdist_wheel', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

if force_setuptools:
    import setuptools

# Import distutils _after_ potential setuptools import above, and after
# removing MANIFEST
from distutils.core import setup
from distutils.extension import Extension

from cythexts import cyproc_exts, get_pyx_sdist
from setup_helpers import (install_scripts_bat, add_flag_checking,
                           SetupDependency, read_vars_from,
                           make_np_ext_builder)
from version_helpers import get_comrec_build

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
using_setuptools = 'setuptools' in sys.modules
extra_setuptools_args = {}
if using_setuptools:
    # Try to preempt setuptools monkeypatching of Extension handling when Pyrex
    # is missing.  Otherwise the monkeypatched Extension will change .pyx
    # filenames to .c filenames, and we probably don't have the .c files.
    sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
    # Set setuptools extra arguments
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        zip_safe=False,
        extras_require=dict(
            doc=['Sphinx>=1.0'],
            test=['nose>=0.10.1']))

# Define extensions
EXTS = []

# We use some defs from npymath, but we don't want to link against npymath lib
ext_kwargs = {'include_dirs': ['src']}  # We add np.get_include() later

for modulename, other_sources, language in (
        ('thetan.cythonutils', [], 'c'),
        ('thetan.theta', [], 'c')):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename, [pyx_src] + other_sources,
                          language=language,
                          **deepcopy(ext_kwargs)))  # deepcopy lists

# Do our own build and install time dependency checking. setup.py gets called
# in many different ways, and may be called just to collect information
# (egg_info). We need to set up tripwires to raise errors when actually doing
# things, like building, rather than unconditionally in the setup.py import or
# exec We may make tripwire versions of build_ext, build_py, install
need_cython = True
pybuilder = get_comrec_build('thetan')
# Cython is a dependency for building extensions, iff we don't have stamped
# up pyx and c files.
build_ext, need_cython = cyproc_exts(EXTS,
                                     info.CYTHON_MIN_VERSION,
                                     'pyx-stamps')
# Add openmp flags if they work
simple_test_c = """int main(int argc, char** argv) { return(0); }"""
omp_test_c = """#include <omp.h>
int main(int argc, char** argv) { return(0); }"""

msc_flag_defines = [[['/openmp'], [], omp_test_c, 'HAVE_VC_OPENMP'],
                    ]
gcc_flag_defines = [[['-msse2', '-mfpmath=sse'], [], simple_test_c, 'USING_GCC_SSE2'],
                    [['-fopenmp'], ['-fopenmp'], omp_test_c, 'HAVE_OPENMP'],
                    ]

# Test if it is a 32 bits version
if not sys.maxsize > 2 ** 32:
    # This flag is needed only on 32 bits
    msc_flag_defines += [[['/arch:SSE2'], [], simple_test_c, 'USING_VC_SSE2'], ]

flag_defines = msc_flag_defines if 'msc' in platform.python_compiler().lower() else gcc_flag_defines

extbuilder = add_flag_checking(build_ext, flag_defines, 'thetan')

# Use ext builder to add np.get_include() at build time, not during setup.py
# execution.
extbuilder = make_np_ext_builder(extbuilder)
if need_cython:
    SetupDependency('Cython', info.CYTHON_MIN_VERSION,
                    req_type='install_requires',
                    heavy=True).check_fill(extra_setuptools_args)
SetupDependency('numpy', info.NUMPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('scipy', info.SCIPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)

cmdclass = dict(
    build_py=pybuilder,
    build_ext=extbuilder,
    install_scripts=install_scripts_bat,
    sdist=get_pyx_sdist(include_dirs=['src']))

setup(
    name=info.__packagename__,
    version=versioneer.get_version(),
    cmdclass=cmdclass,  # versioneer.get_cmdclass(),
    description=info.__description__,
    long_description=info.__longdesc__,
    author=info.__author__,
    author_email=info.__email__,
    maintainer=info.__maintainer__,
    maintainer_email=info.__email__,
    url=info.__url__,
    license=info.__license__,
    packages=['thetan', 'thetan.tests'],
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'thetan': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            ]
        },
    ext_modules=EXTS,
    setup_requires=info.SETUP_REQUIRES,
    # install_requires=info.REQUIRES,
    # tests_require=info.TESTS_REQUIRES,
    # extras_require=info.EXTRA_REQUIRES,
    dependency_links=info.LINKS_REQUIRES,
    classifiers=info.CLASSIFIERS,
    download_url=info.DOWNLOAD_URL,
    provides=info.PROVIDES,
    # zip_safe=False,
    **extra_setuptools_args,
)
