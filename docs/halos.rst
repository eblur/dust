
astrodust.halos
===============

This module is for calculating X-ray surface brightness profiles for dust scattering
halos from both Galactic and cosmological (z > 0) point sources.

Halo class
----------

The **Halo** class is used for both Galactic and extragalactic
sources and contains the attribute *htype*, which can be filled with
one of two objects: either *CosmHalo* (for extragalactic sources) or
*GalHalo* (for Galactic sources, see :doc:`galhalo`).

.. autoclass:: astrodust.halos.Halo

Halo types (*htype*)
--------------------

*CosmHalo*

.. autoclass:: astrodust.halos.cosmhalo.CosmHalo

*GalHalo*

.. autoclass:: astrodust.halos.galhalo.GalHalo


Sub-modules
-----------

.. toctree::
   :maxdepth: 1

   galhalo
   cosmhalo
   cosmology
   halodict
   analytic
   model_halo
