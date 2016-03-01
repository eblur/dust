
sigma_scat
==========

This module combines the complex index of refraction (:doc:`cmindex`)
and scattering algorithm (:doc:`scatmodels`) to calculate the total or
differential cross-sections for a particular dust grain size
distribution (:doc:`dust`).

Classes
-------

The **Scatmodel** object specifies what type of scattering physics will
be used both through the complex index of refraction and the
scattering physics algorithm.

.. autoclass:: sigma_scat.Scatmodel

The remaining object classes contain cross-sections that are
integrated over a particular dust grain size distribution.

.. autoclass:: sigma_scat.Diffscat
.. autoclass:: sigma_scat.Sigmascat
.. autoclass:: sigma_scat.Sigmaext
.. autoclass:: sigma_scat.Kappascat
.. autoclass:: sigma_scat.Kappaext

Functions
---------

The *makeScatmodel* function is a short-cut for creating a **Scatmodel**
object based on two input strings (which become the *stype* and
*cmtype*).

.. autofunction:: sigma_scat.makeScatmodel





