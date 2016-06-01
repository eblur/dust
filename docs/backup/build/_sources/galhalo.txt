
galhalo
=======

This module is for calculating intensity profiles for dust scattering
halos from Galactic point sources

Classes
-------

When creating a **Halo** object, one must import from :doc:`halo`.
The **Halo** object is used for both Galactic and extragalactic
sources and contains the attribute *htype*, which can be filled with
one of two objects: either *CosmHalo* (for extragalactic sources, see
:doc:`halo`) or *GalHalo* (for Galactic sources).

.. autoclass:: galhalo.GalHalo

In addition, Galactic halos are self-similar, which might be useful
for something.

.. autoclass:: galhalo.Ihalo

Functions
---------

Functions within this module modify the **Halo** object by updating
the *htype*, *intensity*, and any other related attributes.
Intensities are calculated using numerical integration (trapezoidal
method).  For the semi-analytic solutions, see :doc:`analytic`.

.. autofunction:: galhalo.UniformISM
.. autofunction:: galhalo.DiscreteISM

There are a number of other functions for calculating useful things.

.. autofunction:: galhalo.path_diff
.. autofunction:: galhalo.make_Ihalo_dict




