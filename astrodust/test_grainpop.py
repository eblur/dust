import pytest
import numpy as np

from astrodust.distlib.grainpop import GrainPop
from astrodust.distlib import Grain, Powerlaw, WD01

# Just test to see if everything runs

@pytest.mark.parametrize('sizedists',[Grain(), Powerlaw(), WD01()])
def test_GrainPop_sizedist(sizedists):
    gp = GrainPop(sizedist=sizedists, composition='Graphite', scatmodel='Mie', md=1.e-4)
    assert isinstance(gp.a, np.ndarray)  # has a size
    assert isinstance(gp.ndens, np.ndarray)  # has a number density
    return

@pytest.mark.parametrize(('compositions','rhos'),
                         ([('Graphite', 2.2),
                           ('Silicate', 3.8),
                           ('Drude', 3.0)]))
def test_GrainPop_composition(compositions, rhos):
    gp = GrainPop(sizedist=Grain(), composition=compositions, scatmodel='Mie')
    assert gp.rho == rhos
    gp = GrainPop(sizedist=Powerlaw(), composition=compositions, scatmodel='Mie')
    assert gp.rho == rhos
    gp = GrainPop(sizedist=WD01(), composition=compositions, scatmodel='Mie')
    assert gp.rho == rhos
    return

wavel = np.logspace(0.0, 1.5, 2)  # Angstroms
TEST_GP = GrainPop(sizedist=Grain(), composition='Graphite', scatmodel='Mie')

# Extinction cross section can only be calculated for Mie
def test_GrainPop_tauext():
    TEST_GP.calc_tau_ext(wavel)
    assert isinstance(TEST_GP.tau_ext, np.ndarray)
    return

def test_GrainPop_tausca():
    TEST_GP.calc_tau_sca(wavel)
    assert isinstance(TEST_GP.tau_sca, np.ndarray)

def test_GrainPop_tauabs():
    TEST_GP.calc_tau_abs(wavel)
    assert isinstance(TEST_GP.tau_abs, np.ndarray)
    new_gp = GrainPop(sizedist=Grain(), composition='Graphite', scatmodel='Mie')
    new_gp.calc_tau_abs(wavel)
    assert isinstance(new_gp.tau_abs, np.ndarray)
