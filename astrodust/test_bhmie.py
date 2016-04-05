"""Test the bhmie module"""

from astrodust.extinction.bhmie import *
from astrodust.distlib.composition import cmindex as cmi

A1 = 0.1  # um
E1 = 1.0  # keV
bhmobj = BHmie(A1, E1, cmi.CmGraphite())

#def test_one_grainsize():
#
