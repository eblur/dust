import numpy as np

from .sizedist import *

"""
    API for GrainPop class
    ----------------------

    | **GrainPop** class contains the following
    |
    | **ATTRIBUTES**
    | sizedist : distlib.Grain(), distlib.Powerlaw(), distlib.WD01()
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
        self.composition = composition
        self.scatmodel = scatmodel

    @property
    def a(self):
        return self.sizedist.a

    def ndens(self, md):
        return self.sizedist.ndens(md)
