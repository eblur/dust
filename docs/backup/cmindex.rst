
cmindex
=======

This module contains built-in complex indices of refraction typically
used for astrophysical dust.

Users wishing to create their own index of refraction should adhere to
the format:

*class* **CmIndex** (object)

    | **ATTRIBUTES**
    | cmtype : string ('Drude', 'Graphite', or 'Silicate')
    | rp     : either a function or scipy.interp1d object that can called, 
    |          e.g. rp(E) where E is in [keV]
    | ip     : same as above, ip(E) where E is in [keV]

.. automodule:: cmindex

Classes
-------

.. autoclass:: cmindex.CmDrude
.. autoclass:: cmindex.CmGraphite
.. autoclass:: cmindex.CmSilicate

Functions
---------

.. autofunction:: cmindex.getCM






