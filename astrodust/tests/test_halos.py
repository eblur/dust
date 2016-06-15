"""Test halos library imports and types."""

import numpy as np
from astrodust.halos import *
from astrodust import grainpop

def test_Halo():
    halo = Halo(1.0)
    assert type(halo) == Halo
    assert type(halo.gpop) == grainpop.GrainPop

# Test the analytic halo module
def test_gammainc():
    assert analytic.gammainc_fun(0,-1) is None

def test_analytic_uniform_funs():
    halo = Halo(1.0)
    analytic.set_htype(halo)
    assert np.size(analytic.G_p(halo)) == 1
    assert type(analytic.G_u(halo)) == np.ndarray
    analytic.uniform_eq(halo)

def test_analytic_screen_funs():
    halo = Halo(1.0)
    analytic.set_htype(halo, xg=0.5)
    assert np.size(analytic.G_p(halo)) == 1
    assert isinstance(analytic.G_s(halo), np.ndarray)
    analytic.screen_eq(halo)

# Test the galhalo module
def test_gh_types():
    assert isinstance(galhalo.GalHalo(), galhalo.GalHalo)

def test_gh_funcs():
    galhalo.path_diff(100.0, 0.5)

def test_gh_uniformISM():
    halo = Halo(1.0)
    galhalo.uniformISM(halo)
    assert (halo.taux is None) is False
    assert isinstance(halo.intensity, np.ndarray)

def test_gh_screenISM():
    halo = Halo(1.0)
    galhalo.screenISM(halo)
    assert (halo.taux is None) is False
    assert isinstance(halo.intensity, np.ndarray)

# Test the cosmological halo module
def test_cosmspec():
    assert isinstance(cosmology.cosm_gpop(), grainpop.GrainPop)

def test_ch_types():
    assert isinstance(cosmhalo.CosmHalo(), cosmhalo.CosmHalo)

def test_ch_uniformIGM():
    halo = Halo(1.0, gpop=cosmology.cosm_gpop())
    cosmhalo.uniformIGM(halo)
    assert (halo.taux is None) is False
    assert isinstance(halo.intensity, np.ndarray)

def test_ch_screenIGM():
    halo = Halo(1.0, gpop=cosmology.cosm_gpop())
    cosmhalo.screenIGM(halo)
    assert (halo.taux is None) is False
    assert isinstance(halo.intensity, np.ndarray)
