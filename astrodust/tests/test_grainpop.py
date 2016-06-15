import pytest
import numpy as np

from astrodust.grainpop import *
from astrodust.distlib import *
from astrodust.extinction import *

DEFAULT_COMP = Composition('Silicate')

# Just test to see if everything runs

@pytest.mark.parametrize('sizedists', [Grain(), Powerlaw(), WD01()])
def test_GrainPop_sizedist(sizedists):
    gp = GrainPop(sizedist=sizedists, composition=DEFAULT_COMP, scatmodel=Mie(), md=1.e-4)
    assert isinstance(gp.a, np.ndarray)  # has a size
    assert isinstance(gp.ndens, np.ndarray)  # has a number density
    return

@pytest.mark.parametrize(('compositions','rhos'),
                         ([(Composition('Graphite'), 2.2),
                           (Composition('Silicate'), 3.8),
                           (Composition('Drude'), 3.0)]))
def test_GrainPop_composition(compositions, rhos):
    gp = GrainPop(sizedist=Grain(), composition=compositions, scatmodel=Mie())
    assert gp.rho == rhos
    gp = GrainPop(sizedist=Powerlaw(), composition=compositions, scatmodel=Mie())
    assert gp.rho == rhos
    gp = GrainPop(sizedist=WD01(), composition=compositions, scatmodel=Mie())
    assert gp.rho == rhos
    return

wavel     = np.logspace(0.0, 1.5, 2)  # Angstroms
one_wavel = 4500.0  # Angstroms
TEST_GP = GrainPop(sizedist=Grain(), composition=DEFAULT_COMP, scatmodel=Mie())

# Extinction cross section can only be calculated for Mie
@pytest.mark.parametrize('wavels', [wavel, one_wavel])
def test_GrainPop_tauext(wavels):
    TEST_GP.calc_tau_ext(wavels)
    assert isinstance(TEST_GP.tau_ext, np.ndarray)
    return

@pytest.mark.parametrize('wavels', [wavel, one_wavel])
def test_GrainPop_tausca(wavels):
    TEST_GP.calc_tau_sca(wavels)
    assert isinstance(TEST_GP.tau_sca, np.ndarray)

@pytest.mark.parametrize('wavels', [wavel, one_wavel])
def test_GrainPop_tauabs(wavels):
    TEST_GP.calc_tau_abs(wavels)
    assert isinstance(TEST_GP.tau_abs, np.ndarray)
    new_gp = GrainPop(sizedist=Grain(), composition=DEFAULT_COMP, scatmodel=Mie())
    new_gp.calc_tau_abs(wavel)
    assert isinstance(new_gp.tau_abs, np.ndarray)
