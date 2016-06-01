
analytic
========

This module contains semi-analytic functions for calculating X-ray
scattering halo intensities from a power-law distribution of dust
grain sizes (see Appendix of `Corrales & Paerels, 2015
<http://adsabs.harvard.edu/abs/2015MNRAS.453.1121C>`_).


When creating a **Halo** object, one must import from :doc:`halo`.
The functions in this module are for Galactic sources only.  (Note
that :doc:`galhalo` contains functions for numerically integrated
solutions).


Functions
---------

Functions within this module modify the **Halo** object by updating
the *htype* and *intensity*.

.. autofunction:: analytic.screen_eq
.. autofunction:: analytic.uniform_eq





