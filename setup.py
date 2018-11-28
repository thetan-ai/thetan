from os import path
from setuptools import setup, find_packages
import sys
import versioneer
from skepsi.__about__ import (__packagename__, __version__, __author__,
                              __email__, __maintainer__, __license__,
                              __description__, __longdesc__, __url__,
                              DOWNLOAD_URL, CLASSIFIERS, PROVIDES,
                              REQUIRES, SETUP_REQUIRES, LINKS_REQUIRES,
                              TESTS_REQUIRES, EXTRA_REQUIRES,
                            )


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 4)
if sys.version_info < min_version:
    error = """
skepsi does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*sys.version_info[:2], *min_version)
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name=__packagename__,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=__description__,
    long_description=__longdesc__,
    author=__author__,
    author_email=__email__,
    maintainer=__maintainer__,
    maintainer_email=__email__,
    url=__url__,
    license=__license__,
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'skepsi': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            ]
        },
    setup_requires=SETUP_REQUIRES,
    install_requires=REQUIRES,
    tests_require=TESTS_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    dependency_links=LINKS_REQUIRES,
    classifiers=CLASSIFIERS
    download_url=DOWNLOAD_URL,
    provides=PROVIDES,
    # zip_safe=False,
)
