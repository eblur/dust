import numpy as np

from .sizedist import *
from .composition import *

"""
    API for GrainPop class
    ----------------------

    | **GrainPop** class contains the following
    |
    | **ATTRIBUTES**
    | sizedist : distlib.Grain(), distlib.Powerlaw(), distlib.WD01()
    | composition : a string describing the grain composition
    | scatmodel : a string describing the extinction model to use
"""

class GrainPop(object):
    """
    A population of dust grains
        |
        | **ATTRIBUTES***
        | sizedist
        | composition
        | scatmodel
    """
    def __init__(self, sizedist, composition, scatmodel):
        self.sizedist = sizedist
        self.composition = Composition(composition)
        self.scatmodel = scatmodel

    @property
    def a(self):
        return self.sizedist.a

    @property
    def rho(self):
        return self.composition.rho

    def ndens(self, md):
        return self.sizedist.ndens(md)
