sigma_scat
==========

This module combines the complex index of refraction (:doc:`cmindex`)
and scattering algorithm (:doc:`scatmodels`) to calculate the total or
differential cross-sections for a particular dust grain size
distribution (:doc:`dust`).

Classes
-------

The **ScatModel** object specifies what type of scattering physics will
be used both through the complex index of refraction and the
scattering physics algorithm.

.. autoclass:: astrodust.extinction.sigma_scat.ScatModel

The remaining classes contain cross-sections that are
integrated over a particular dust grain size distribution.

.. autoclass:: astrodust.extinction.sigma_scat.DiffScat
.. autoclass:: astrodust.extinction.sigma_scat.SigmaScat
.. autoclass:: astrodust.extinction.sigma_scat.SigmaExt
.. autoclass:: astrodust.extinction.sigma_scat.KappaScat
.. autoclass:: astrodust.extinction.sigma_scat.KappaExt

Functions
---------

The *makeScatmodel* function is a short-cut for creating a **Scatmodel**
object based on two input strings (which become the *stype* and
*cmtype*).

.. autofunction:: astrodust.extinction.sigma_scat.makeScatModel
