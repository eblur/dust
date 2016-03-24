"""Test the distlib."""

import distlib

def test_grain():
    assert type(distlib.Grain()) == distlib.Grain

def test_Powlaw():
    assert type(distlib.Powerlaw()) == distlib.Powerlaw

def test_DustSpectrum():
    assert type(distlib.DustSpectrum()) == distlib.DustSpectrum

def test_MRN_dist():
    assert type(distlib.MRN_dist()) == distlib.DustSpectrum
