
model_halo
==========

:doc:`index`

This module is for modeling scattering halos for Galactic sources (see
:doc:`galhalo`) from a dust grain size distribution (:doc:`dust`) and
apparent flux spectrum of a point source.  It uses **HaloDict**
objects from the :doc:`halodict` module.

Functions
---------

.. autofunction:: model_halo.screen
.. autofunction:: model_halo.uniform

Note that this function uses **halodict.HaloDict.total_halo**

.. autofunction:: totalhalo

.. autofunction:: model_halo.simulate_intensity
.. autofunction:: model_halo.simulate_screen
.. autofunction:: model_halo.simulate_uniform



