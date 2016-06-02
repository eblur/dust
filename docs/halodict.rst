
halodict
========

This module is for creating and manipulating **HaloDict** objects (a
dictionary of **Halo** objects, see :doc:`halos`).  The keys of the
**HaloDict** objects are energy values.

Classes
-------

.. autoclass:: astrodust.halos.halodict.HaloDict
    :members: fitsify_halodict

Functions
---------

Function for reading in **HaloDict** objects that have been saved to a FITS
file via **HaloDict.fitsify_halodict()**

.. autofunction:: astrodust.halos.halodict.read_halodict_fits
