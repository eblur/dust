"""Test extinction library"""

import numpy as np

from astrodust.extinction import *

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

## Test cross section models

def test_sigmascat_makeScatModel():
    assert type(sigma_scat.makeScatModel('RG','Drude')) == sigma_scat.ScatModel
    assert type(sigma_scat.makeScatModel('RG','Silicate')) == sigma_scat.ScatModel
    assert type(sigma_scat.makeScatModel('RG','Graphite')) == sigma_scat.ScatModel
    assert type(sigma_scat.makeScatModel('Mie','Silicate')) == sigma_scat.ScatModel
    assert type(sigma_scat.makeScatModel('Mie','Graphite')) == sigma_scat.ScatModel

def test_sigmscat_rgdrude():
    RGD = sigma_scat.makeScatModel('RG','Drude')
    assert RGD.smodel.stype is 'RGscat'
    assert RGD.cmodel.cmtype is 'Drude'

def test_sigmascat_classes():
    assert type(sigma_scat.DiffScat()) == sigma_scat.DiffScat
    assert type(sigma_scat.SigmaScat()) == sigma_scat.SigmaScat
    assert type(sigma_scat.SigmaExt()) == sigma_scat.SigmaExt
