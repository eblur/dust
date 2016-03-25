"""Test grainmodel library"""

## Supporting modules
import numpy as np
from scipy.interpolate import interp1d
from dust import distlib

## The main module being tested
from dust.distlib import grainmodel

# Set up a dust grain distribution object
# Should contain grain radius, number density, and a function for computing
#  absolute number density given A_V, NH and d2g ratio, or M_d
def test_GrainDist():
    gdist = grainmodel.GrainDist()
    # Grain distribution radius
    assert type(gdist).a)  = np.ndarry
    assert type(gdist.nd) = np.ndarry

    # there should be a "distribution type" object that contains metadata,
    # e.g. powerlaw, WD01, or ZDA model types
    assert type(gdist.dtype.name) == string
    assert type(gdist.dtype) in [distlib.Grain, distlib.Powerlaw, distlib.WD01]
    # ^ Right now WD01 needs to be implemented differently to satisfy this

    # there should be functions that regrids the grain radius
    new_a = np.logspace(-2, -1, 25)
    gdist.regrid(new_a)
    assert len(gdist.a) == len(new_a)

    # there should be functions that re-calculate the number density
    #  given a normalization factor
    gdist.renorm_AV(av)
    gdist.renorm_NH(NH, d2g=0.009)
    gdist.renorm_md(md)

    # Composition includes a name and complex index of refraction
    assert type(gdist.comp.name) in ['Silicate','Graphite','Drude']
    assert type(gidst.comp.cm.rp) == scipy.interp1d
    assert type(gidst.comp.cm.ip) == scipy.interp1d

    # Shape contains parameters describing its geometry
    # Right now only spheres are supported
    assert gdist.shape.name == 'sphere'
    # geometric cross-section (cgeo) is pi*a^2 for spherical grains
    assert type(gdist.shape.cgeo) == np.ndarray


# At some point, GrainModel will be a constainer for:
# - size distributions (GrainDist)
# - extinction models
# - thermal emission stuff
#
def test GrainModel():
    gmodel = grainmodel.GrainModel()
    assert type(gmodel) = grainmodel.GrainModel
    assert type(gmodel.gdist) = grainmodel.GrainDist
