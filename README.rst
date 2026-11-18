
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

To run the example please go to thetan/docs/examples

.. code-block::

    python example.py


