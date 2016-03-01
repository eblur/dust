
halo
====

This module is for calculating intensity profiles for dust scattering
halos from cosmological (z > 0) point sources.

Classes
-------

The **Halo** object is used for both Galactic and extragalactic
sources and contains the attribute *htype*, which can be filled with
one of two objects: either *CosmHalo* (for extragalactic sources) or
*GalHalo* (for Galactic sources, see :doc:`galhalo`).

.. autoclass:: halo.Halo

Object definition for the halo type *CosmHalo*

.. autoclass:: halo.CosmHalo

Functions
---------

Functions within this module modify the **Halo** object by updating
the *htype*, *intensity*, and any other related attributes.

.. autofunction:: halo.UniformIGM
.. autofunction:: halo.ScreenIGM






