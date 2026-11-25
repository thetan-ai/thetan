
Thetan
=========
Innovative algorithms for the AI era

Installation
============

We assume you have recent Anaconda distribution with scipy, numpy, cython and matplotlib.

For the cython parts of the code to compile you will need a compiler such as GCC or VC++.

To Install the packages and create module, compile using the following command.

.. code-block::

    pip install -r requirements/build.txt
    pip install --no-build-isolation  .

This installs thetan with numpy as its only requirement. ``tbscan``,
``thetan.accuracy``, ``thetan.patchify`` and ``thetan.viz`` also need scipy,
scikit-learn, scikit-image and matplotlib. To install them, use

.. code-block::

    pip install -r requirements/optional.txt

Examples live in ``docs/examples``. Run one with

.. code-block::

    python docs/examples/berserker_gaussian.py

Website
=======

The website (landing page, guides, API reference and example gallery) is a
`MyST <https://mystmd.org>`_ project in ``docs/``. It needs Node.js 18 or
later. Build and serve it with

.. code-block::

    pip install -r requirements/docs.txt
    spin docs --serve

then open http://127.0.0.1:8000/.


