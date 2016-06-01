
composition.cmindex
===================

This module contains built-in complex indices of refraction typically
used for astrophysical dust.

API for CmIndex classes
-----------------------

*class* **CmIndex** (object)

    | **ATTRIBUTES**
    | cmtype : string ('Drude', 'Graphite', or 'Silicate')
    | rp     : either a function or scipy.interp1d object that can called,
    |          e.g. rp(E) where E is in [keV]
    | ip     : same as above, ip(E) where E is in [keV]

.. automodule:: cmindex

Classes
-------

.. autoclass:: astrodust.distlib.composition.cmindex.CmDrude
.. autoclass:: astrodust.distlib.composition.cmindex.CmGraphite
.. autoclass:: astrodust.distlib.composition.cmindex.CmSilicate

Functions
---------

.. autofunction:: astrodust.distlib.composition.cmindex.getCM


composition.minerals
====================
