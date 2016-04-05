"""Test the bhmie module"""

import pytest
import numpy as np

from astrodust.extinction.bhmie import *
from astrodust.distlib.composition import cmindex as cmi

A1 = 0.1  # um
E1 = 1.0  # keV
ALIST = np.linspace(0.05, 0.3, 10)
ELIST = np.logspace(-1, 1, 20)

@pytest.mark.parametrize('vals',[[A1, E1],[A1,ELIST],[ALIST,E1],[ALIST,ELIST]])
def test_bhmie_instance(vals):
    a, e = vals
    bhobj = BHmie(a, e, cmi.CmSilicate())
    assert bhobj.E.shape == bhobj.a.shape

'''def test_one_grainsize():
    bhobj = BHmie(A1, ELIST, cmi.CmSilicate())
    bhobj.calculate()'''
