"""Test grainmodel library"""

import numpy as np
from dust import distlib.grainmodel

# Set up a dust grain distribution object
# Should contain grain radius, number density, and a function for computing
#  absolute number density given A_V, NH and d2g ratio, or M_d
def test_GrainDist():
    gdist = grainmodel.GrainDist()
    # Grain distribution radius
    assert type(GrainDist().a)  = np.ndarry
    assert type(GrainDist().nd) = np.ndarry
    # there should be a "distribution type" object that contains metadata,
    # e.g. powerlaw, WD01, or ZDA model types
    gdist.dtype
    # there should be functions that regrids the grain radius
    gdist.regrid(new_a)

def test GrainModel():
    gmodel = grainmodel.GrainModel()
    assert type(gmodel) = grainmodel.GrainModel
