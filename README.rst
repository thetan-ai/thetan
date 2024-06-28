
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

    process_sims.py
    process_superpixel.py
    process_t1.py
    process_ao.py
    process_ecg.py
    process_t1_slices.py

To Find the dataset used in the examples, download the dataset from the following link:

- ECG dataset: https://physionet.org/content/ptb-xl/1.0.3/
- IXI dataset: https://brain-development.org/ixi-dataset/

NOTE
====

- Thetan Berseker is represeented by the function `berserker2` in `thetan/berserker.py`.
- TSG is represented by the function `theta_g1` in `thetan/theta.py`.

