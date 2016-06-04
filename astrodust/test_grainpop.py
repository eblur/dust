import pytest
import numpy as np

from astrodust.distlib.grainpop import GrainPop
from astrodust.distlib import Grain, Powerlaw, WD01

TEST_MD = 1.e-4  # g cm^-2

@pytest.mark.parametrize('sizedists',[Grain(), Powerlaw(), WD01()])
def test_GrainPop_sizedist(sizedists):
    gp = GrainPop(sizedist=sizedists, composition='Graphite', scatmodel='Mie')
    assert isinstance(gp.a, np.ndarray)  # has a size
    assert isinstance(gp.ndens(TEST_MD), np.ndarray)  # has a number density
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
