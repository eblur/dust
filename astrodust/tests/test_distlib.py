"""Test the distlib."""

from dust import distlib

def test_grain():
    assert type(distlib.Grain()) == distlib.Grain

def test_Powlaw():
    assert type(distlib.Powerlaw()) == distlib.Powerlaw

def test_DustSpectrum():
    assert type(distlib.DustSpectrum()) == distlib.DustSpectrum

def test_MRN_dist():
    assert type(distlib.MRN_dist()) == distlib.DustSpectrum

def test_WD01_dist():
    assert type(distlib.make_WD01_DustSpectrum()) == distlib.DustSpectrum

# Test both powerlaw and grain thingies
def test_calc_from_dist():
    pl = distlib.Powerlaw()
    DSp = distlib.DustSpectrum()
    DSp.calc_from_dist(pl, md=1.e-4)
    gr = distlib.Grain()
    DSg = distlib.DustSpectrum()
    DSg.calc_from_dist(gr, md=1.e-4)
