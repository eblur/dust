import pytest
import numpy as np

from astrodust.distlib.grainpop import GrainPop
from astrodust.distlib import Grain, Powerlaw

TEST_MD = 1.e-4  # g cm^-2

@pytest.mark.parametrize('sizedists',[Grain(), Powerlaw()])
def test_Grainpop_sizedist(sizedists):
    gp = GrainPop(sizedist=sizedists, composition='Graphite', scatmodel='Mie')
    assert isinstance(gp.a, np.ndarray)  # has a size
    assert isinstance(gp.ndens(TEST_MD), np.ndarray)  # has a number density
    return
