"""Test extinction library"""
import pytest
import numpy as np

from astrodust.extinction import *
from astrodust.distlib.composition.cmindex import *
from astrodust.distlib import *

# Test initialization of the scattering models

def test_scatmodels_RGscat():
    assert isinstance(scatmodels.RGscat(), scatmodels.RGscat)

def test_scatmodels_Mie():
    assert isinstance(scatmodels.Mie(), scatmodels.Mie)

def test_scatmodels_PAH():
    assert isinstance(scatmodels.PAH('neu'), scatmodels.PAH)
    assert isinstance(scatmodels.PAH('ion'), scatmodels.PAH)

def test_scatmodels_PAH_Qsca():
    PAH_neu = scatmodels.PAH('neu')
    PAH_ion = scatmodels.PAH('ion')
    assert isinstance(PAH_neu.Qsca(1.0, 0.01), np.ndarray)
    assert isinstance(PAH_neu.Qsca(1.0, 0.01), np.ndarray)
    assert isinstance(PAH_ion.Qsca(1.0, 0.01), np.ndarray)
    assert isinstance(PAH_ion.Qsca(1.0, 0.01), np.ndarray)

# Test that the sigma_scat module loades
ATEST = 0.1
ETEST = np.array([0.3,0.5])

@pytest.mark.parametrize(('scatmods','cms'),
                         [(RGscat(),CmDrude()),
                          (RGscat(),CmSilicate()),
                          (RGscat(),CmGraphite()),
                          (Mie(),CmSilicate()),
                          (Mie(),CmGraphite())])
def test_sigma_sca(scatmods, cms):
    test = sigma_sca(ATEST, ETEST, scatm=scatmods, cm=cms)
    assert isinstance(test, np.ndarray)

@pytest.mark.parametrize(('scatmods','cms'),
                         [(Mie(),CmSilicate()),
                          (Mie(),CmGraphite())])
def test_sigma_ext(scatmods, cms):
    test = sigma_ext(ATEST, ETEST, scatm=scatmods, cm=cms)
    assert isinstance(test, np.ndarray)

@pytest.mark.parametrize(('scatmods','cms'),
                         [(RGscat(),CmDrude()),
                          (RGscat(),CmSilicate()),
                          (RGscat(),CmGraphite()),
                          (Mie(),CmSilicate()),
                          (Mie(),CmGraphite())])
def test_kappa_sca(scatmods, cms):
    test = kappa_sca(ETEST, dist=Grain(), scatm=scatmods, cm=cms)
    assert isinstance(test, np.ndarray)

@pytest.mark.parametrize(('scatmods','cms'),
                         [(Mie(),CmSilicate()),
                          (Mie(),CmGraphite())])
def test_kappa_ext(scatmods, cms):
    test = kappa_ext(ETEST, dist=Grain(), scatm=scatmods, cm=cms)
    assert isinstance(test, np.ndarray)

@pytest.mark.parametrize(('scatmods','cms'),
                         [(RGscat(),CmDrude()),
                          (RGscat(),CmSilicate()),
                          (RGscat(),CmGraphite()),
                          (Mie(),CmSilicate()),
                          (Mie(),CmGraphite())])
def test_diff_scat(scatmods, cms):
    AVAL, EVAL, THETA = 0.1, 0.3, np.linspace(5.0, 100.0, 3)
    test = diff_scat(AVAL, EVAL, scatm=scatmods, cm=cms, theta=THETA)
    assert len(test) == len(THETA)
