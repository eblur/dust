"""Test the bhmie module"""

import pytest
import numpy as np

from astrodust.extinction.bhmie import *
from astrodust.distlib.composition import cmindex as cmi

A1 = 0.1  # um
E1 = 1.0  # keV
ALIST = np.linspace(0.05, 0.3, 10)
ELIST = np.logspace(-1, 1, 20)
THLIST = np.logspace(-1, 2, 5)

@pytest.mark.parametrize('vals',[[A1, E1],[A1,ELIST],[ALIST,E1],[ALIST,ELIST]])
def test_bhmie_instance(vals):
    a, e = vals
    bhobj = BHmie(a, e, cmi.CmSilicate())
    assert bhobj.E.shape == bhobj.a.shape

@pytest.mark.parametrize('vals',[[A1, E1],[A1,ELIST],[ALIST,E1],[ALIST,ELIST]])
def test_BHmie_calculate(vals):
    a, e = vals
    print("NA = %d, NE = %d, NTH = 1" % (np.size(a), np.size(e)))
    bhobj = BHmie(a, e, cmi.CmSilicate())
    bhobj.calculate()
    assert bhobj.qsca.shape == bhobj.X.shape
    assert bhobj.S1.shape == (bhobj.NA, bhobj.NE, bhobj.NTH)

@pytest.mark.parametrize('vals',[[A1, E1],[A1,ELIST],[ALIST,E1],[ALIST,ELIST]])
def test_BHmie_calculate_theta(vals):
    a, e  = vals
    print("NA = %d, NE = %d, NTH = %d" %
          (np.size(a), np.size(e), np.size(THLIST)))
    bhobj = BHmie(a, e, cmi.CmSilicate())
    bhobj.calculate(THLIST)
    assert bhobj.qsca.shape == bhobj.X.shape
    assert bhobj.S1.shape == (bhobj.NA, bhobj.NE, bhobj.NTH)
    assert bhobj.diff.shape == (bhobj.NA, bhobj.NE, bhobj.NTH)
