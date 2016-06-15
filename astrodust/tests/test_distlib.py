"""Test the distlib."""
import pytest
import numpy as np
from scipy.integrate import trapz

from astrodust.distlib import *
from astrodust import constants as c

TEST_MD, TEST_RHO = 1.e-4, 4.0

def test_grain():
    test = Grain()
    assert len(test.ndens(TEST_MD, rho=TEST_RHO)) == len(test.a)

def test_Powlaw():
    test = Powerlaw()
    assert len(test.ndens(TEST_MD, rho=TEST_RHO)) == len(test.a)

@pytest.mark.parametrize(('gals','compositions'),
                         [('MW','Silicate'),
                          ('MW','Graphite'),
                          ('LMC','Silicate'),
                          ('LMC','Graphite')])
def test_WD01(gals, compositions):
    test = WD01(comp=compositions, gal=gals)
    assert len(test.ndens(TEST_MD, rho=TEST_RHO)) == len(test.a)

#------------------------------------------
# UNIT TESTS

def test_units():
    assert Grain().ndens(0.0) == 0.0
    assert np.sum(Powerlaw().ndens(0.0)) == 0.0
    assert np.all(np.isinf(Powerlaw().ndens(md=1.0, rho=0.0)))

@pytest.mark.parametrize('gmodels', [Grain(), Powerlaw(), WD01()])
def test_ndens_units(gmodels):
    test = gmodels
    assert np.all(test.ndens(2.0*TEST_MD)/test.ndens(TEST_MD) == 2.0)
    assert np.all(test.ndens(TEST_MD, rho=TEST_RHO/2.0)/test.ndens(TEST_MD, rho=TEST_RHO) == 2.0)

def test_WD01_ndens_units():
    test = WD01(comp='Silicate')
    # No input to ndens yields nominal dust mass column
    mg   = (4./3.) * np.pi * test.comp.rho * np.power(test.a * c.micron2cm, 3)
    assert np.abs(1.0 - trapz(test.ndens()*mg, test.a) / test.md_nom) < 0.01
    # Test that ndens changes appropriately with dust mass column
    md_new = 1.e-3
    assert np.abs(1.0 - trapz(test.ndens(md=md_new)*mg, test.a) / md_new) < 0.01
    assert np.all(test.ndens(md=md_new) > test.nd_nom)
    # Test that changing dust material density (rho) will still yield correct dust mass
    rho_new = 4.0
    mg_new  = (4./3.) * np.pi * rho_new * np.power(test.a * c.micron2cm, 3)
    assert np.abs(1.0 - trapz(test.ndens(rho=rho_new)*mg_new, test.a) / test.md_nom) < 0.01
    assert np.all(test.ndens(rho=rho_new) < test.nd_nom)
    # Test that you get correct dust mass when changing both
    assert np.abs(1.0 - trapz(test.ndens(md=md_new, rho=rho_new)*mg_new, test.a) / md_new) < 0.01
