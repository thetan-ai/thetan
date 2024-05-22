
Thetan
=========
Official library of ThetA (Theta Algorithms)

Installation
============

We assume you have recent Anaconda distribution with scipy, numpy, cython and seaborn.

For the cython parts of the code to compile you will need a compiler such as GCC or VC++.

To Install the packages and create module, compile using the following command.

.. code-block::

    pip install -r requirements/build.txt
    pip install --no-build-isolation  .

To run the examples with ThetA (TSG, TDG) go to thetan/docs/examples

.. code-block::

    python filename.py

    process_ao.py
    process_ecg.py
    process_sims.py
    process_superpixel.py
    process_t1_slices.py
    process_t1.py

To Find the dataset used in the examples, download the dataset from the following link:

- ECG dataset: https://physionet.org/content/ptb-xl/1.0.3/
- IXI dataset: https://brain-development.org/ixi-dataset/

