
scatmodels
==========

This module contains built-in algorithms for Mie scattering and the
Rayleigh-Gans approximation.

Users wishing to create their own scattering functions should adhere
to the following format:

*class* **Scatmodel** (object)

    | **FUNCTIONS**
    | Qsca with inputs  
    | ( E  : scalar or np.array [keV]
    |   cm : cmtype object cmindex.py
    |    a : scalar [grain size, micron] )
    |    *returns* scalar or np.array [scattering efficiency, unitless]
    |
    | Diff with inputs
    |  ( cm : cmtype object from cmindex.py
    | theta : scalar or np.array [angle, arcsec]
    |    a  : scalar [grain size, micron]
    |    E  : scalar or np.array [energy, keV]
    |    ** if len(E) > 1 and len(theta) > 1, then len(E) must equal len(theta)
    |    *returns* dsigma/dOmega of values (E0,theta0), (E1,theta1) etc...   
    |
    | (optional)
    | Qext with inputs
    | ( E  : scalar or np.array [keV]
    |   cm : cmtype object cmindex.py
    |    a : scalar [grain size, micron] )
    |    *returns* scalar or np.array [extinction efficiency, unitless]

Classes
-------

.. autoclass:: scatmodels.RGscat
.. autoclass:: scatmodels.Mie
.. autoclass:: scatmodels.PAH









