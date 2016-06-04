import numpy as np

from .sizedist import *
from .composition import *
from ..extinction import sigma_scat as ss
from .. import constants as c

# Default dust mass column density
DEFAULT_MD = 1.e-4  # g cm^-2

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
    def __init__(self, sizedist, composition, scatmodel, md=DEFAULT_MD):
        self.sizedist    = sizedist
        self.composition = Composition(composition)
        self.scatmodel   = ss.makeScatModel(scatmodel, composition)
        self.md          = md

    @property
    def a(self):
        return self.sizedist.a

    @property
    def rho(self):
        return self.composition.rho

    @property
    def ndens(self):
        return self.sizedist.ndens(self.md)

    def calc_tau_ext(self, wavel):
        """
        Calculate total extinction cross section for dust population
            | **INPUTS**
            | wavel : ndarray : Wavelengths to use [Angstroms]
            |
            | **RETURNS**
            | \tau_{ext} = \int \sigma_ext(a) dn/da da
        """
        E_keV = c.hc_angs / wavel  # keV
        kappa = ss.kappa_ext(E_keV, scatm=self.scatmodel,
                             dist=self.sizedist, md=self.md)
        return kappa * self.md

    def calc_tau_sca(self, wavel):
        """
        Calculate total scattering cross section for dust population
            | **INPUTS**
            | wavel : Wavelengths to use [Angstroms]
            |
            | **RETURNS**
            | \tau_{sca} = \int \sigma_sca(a) dn/da da
        """
        E_keV = c.hc_angs / wavel  # keV
        kappa = ss.kappa_scat(E_keV, scatm=self.scatmodel,
                              dist=self.sizedist, md=self.md)
        return kappa * self.md
