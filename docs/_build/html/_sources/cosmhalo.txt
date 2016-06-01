cosmhalo
========

This module is for calculating X-ray scattering halos from
Galactic (z=0) point sources.

One must import the **Halo** object from :doc:`halos`.
The **Halo** object is used for both Galactic and extragalactic
sources and contains the attribute *htype*, which can be filled with
one of two objects: either *CosmHalo* (for extragalactic sources)
or *GalHalo* (for Galactic sources).  See :doc:`halos` for a
description of these halo types.

Functions
---------

Functions within this module take a **Halo** object as input and modifies it by updating
the *htype*, *intensity*, and any other related attributes.

.. autofunction:: astrodust.halos.cosmhalo.uniformIGM
.. autofunction:: astrodust.halos.cosmhalo.screenIGM
