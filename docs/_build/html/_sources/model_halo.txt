
model_halo
==========

This module is for modeling scattering halos for Galactic sources (see
:doc:`galhalo`) from a dust grain size distribution (:doc:`dust`) and
apparent flux spectrum of a point source.  It uses **HaloDict**
objects from the :doc:`halodict` module.

Functions
---------

.. autofunction:: model_halo.screen
.. autofunction:: model_halo.uniform

The following two functions are helpers for integrating up the total
intensity from a **HaloDict** object.

.. autofunction:: model_halo.totalhalo
.. autofunction:: model_halo.simulate_intensity

The following functions take inputs that will allow us to simulate a
scattering halo intensity profile from a set of basic starting
parameters.

.. autofunction:: model_halo.simulate_screen
.. autofunction:: model_halo.simulate_uniform



