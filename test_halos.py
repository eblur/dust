"""Test halos library imports and types."""

import numpy as np
from halos import *

def test_Halo():
    halo = Halo(1.0)
    assert type(halo) == Halo
    halo.energy


## Test the galhalo module
def test_gh_types():
    assert type(galhalo.GalHalo()) == galhalo.GalHalo
    assert type(galhalo.Ihalo()) == galhalo.Ihalo

def test_gh_funcs():
    galhalo.path_diff(100.0, 0.5)

def test_gh_uniformISM():
    halo = Halo(1.0)
    galhalo.uniformISM(halo)
    assert (halo.taux is None) == False
    assert type(halo.intensity) == np.ndarray

def test_gh_screenISM():
    halo = Halo(1.0)
    galhalo.screenISM(halo)
    assert (halo.taux is None) == False
    assert type(halo.intensity) == np.ndarray

## Test the cosmological halo module
