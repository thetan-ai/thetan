""" This file contains defines parameters for skepsi that we use to fill
settings in setup.py, the skepsi top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import skepsi
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha', 'Natural Language :: English',
    'Programming Language :: Python :: 3', 'Environment :: Console',
    'Intended Audience :: Science/Research', 'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    ]

description = 'Skepsi: Python package for advanced artificial intelligence'

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """========================================================
Skepsi: Python package for advanced artificial intelligence
========================================================
Skepsi combine Reinforcement Learning, Machine Learning and computionnal
anatony technique to create a unique environment to ease your research on 
this area.
"""

# versions
CYTHON_MIN_VERSION = '0.25.1'
NUMPY_MIN_VERSION = '1.7.1'
SCIPY_MIN_VERSION = '0.9'
NIBABEL_MIN_VERSION = '2.3.0'
DIPY_MIN_VERSION = '0.14.0'
PYTEST_MIN_VERSION = '3.0'

__packagename__ = 'skepsi'
__author__ = __maintainer__ = 'Garyfallidis Research Group (GRG) developers'
__email__ = 'garyfallidis@gmail.com'
__license__ = 'BSD (3-clause)'
__status__ = 'Pre-Alpha'
__description__ = description
__longdesc__ = long_description
__url__ = 'https://gitlab.com/grg_edge/skepsi.git'

DOWNLOAD_URL = ('https://gitlab.com/grg_edge/skepsi/{name}/archives/{ver}.tar.gz'.format(
    name=__packagename__, ver=__version__))
PLATFORMS = 'OS Independent'
MAJOR = __version__.split('.')[0]
MINOR = __version__.split('.')[1]
MICRO = __version__.replace('-', '.').split('.')[2]
ISRELEASE = (len(__version__.replace('-', '.').split('.')) == 3
             or 'post' in __version__.replace('-', '.').split('.')[-1])
VERSION = __version__
PROVIDES = ['skepsi']
REQUIRES = [
    'cython>=%s' % CYTHON_MIN_VERSION,
    'numpy>=%s' % NUMPY_MIN_VERSION,
    'scipy>=%s' % SCIPY_MIN_VERSION,
    'nibabel>=%s' % NIBABEL_MIN_VERSION,
    'dipy>=%s' % DIPY_MIN_VERSION,
    'pytest>=%s' % PYTEST_MIN_VERSION,
]

SETUP_REQUIRES = ['setuptools>=27.0']
TESTS_REQUIRES = ['pytest-cov', 'codecov', 'pytest-env', 'pytest-xdist']
LINKS_REQUIRES = []

EXTRA_REQUIRES = {
    'tests': TESTS_REQUIRES,
    'dev': TESTS_REQUIRES,
    'plugins': [],
}


def _list_union(iterable):
    return list(set(sum(iterable, [])))


# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = _list_union(EXTRA_REQUIRES.values())
