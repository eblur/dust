"""Test extinction library"""

import numpy as np

from .extinction import *

## Test the scattering models

def test_scatmodels_RGscat():
    assert type(scatmodels.RGscat()) == scatmodels.RGscat

def test_scatmodels_Mie():
    assert type(scatmodels.Mie()) == scatmodels.Mie

def test_scatmodels_PAH():
    assert type(scatmodels.PAH('neu')) == scatmodels.PAH
    assert type(scatmodels.PAH('ion')) == scatmodels.PAH

def test_scatmodels_PAH_Qsca():
    PAH_neu = scatmodels.PAH('neu')
    PAH_ion = scatmodels.PAH('ion')
    assert type(PAH_neu.Qsca(1.0, 0.01)) == np.ndarray
    assert type(PAH_neu.Qsca(1.0, 0.01)) == np.ndarray
