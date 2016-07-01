import numpy as np

from .distlib import *
from .extinction import *
from . import constants as c

__all__ = ['GrainPop', 'make_MRN_grainpop']

# Default dust mass column density
DEFAULT_MD = 1.e-4  # g cm^-2

class GrainPop(object):
    """
    A population of dust grains
        |
        | **INITIALIZE**
        | sizedist : grain size dist objects : distlib.Grain, distlib.Powerlaw, distlib.WD01
        | composition : Composition object : Composition('Graphite', 'Silicate' or 'Drude')
        | scatmodel : scatmodel object : extinction.RGscat or extinction.Mie
        | md : dust mass column : 1.e-4 g cm^-2 (default)
        |
        | **ATTRIBUTES**
        | sizedist : distlib.Grain(), distlib.Powerlaw(), distlib.WD01()
        | comp : distlib.Composition class
        | scatm : scatmodel object
        | md : dust mass column [g cm^-2]
        | ext : extinction.ExtinctionCurve object
    """
    def __init__(self, sizedist, composition, scatmodel, md=DEFAULT_MD):
        self.sizedist = sizedist
        self.comp     = composition
        self.scatm    = scatmodel
        self.md       = md
        self.ext      = ExtinctionCurve()

    @property
    def a(self):
        return self.sizedist.a

    @property
    def rho(self):
        return self.comp.rho

    @property
    def ndens(self):
        return self.sizedist.ndens(md=self.md, rho=self.comp.rho)

    @property
    def tau_ext(self):
        return self.ext.tau_ext

    @property
    def tau_sca(self):
        return self.ext.tau_sca

    @property
    def tau_abs(self):
        return self.ext.tau_abs

    def calc_tau_ext(self, wavel):
        """
        Calculate total extinction cross section for dust population
            | **INPUTS**
            | wavel : ndarray : Wavelengths to use [Angstroms]
            |
            | **RETURNS**
            | \tau_{ext} = \int \sigma_ext(a) dn/da da
        """
        # If an extinction calculation has already occured, check to see if
        #  it's the same grid and print a warning if it is not.
        if self.ext.wavel is not None:
            _test_wavel_grid(wavel, self)

        E_keV = c.hc_angs / wavel  # keV
        kappa = kappa_ext(E_keV, scatm=self.scatm, cm=self.comp.cmindex, dist=self.sizedist)
        self.ext.wavel   = wavel
        self.ext.energy  = E_keV
        self.ext.tau_ext = kappa * self.md
        return

    def calc_tau_sca(self, wavel):
        """
        Calculate total scattering cross section for dust population
            | **INPUTS**
            | wavel : Wavelengths to use [Angstroms]
            |
            | **RETURNS**
            | \tau_{sca} = \int \sigma_sca(a) dn/da da
        """
        # If an extinction calculation has already occured, check to see if
        #  it's the same grid and print a warning if it is not.
        if self.ext.wavel is not None:
            _test_wavel_grid(wavel, self)

        E_keV = c.hc_angs / wavel  # keV
        kappa = kappa_sca(E_keV, scatm=self.scatm, cm=self.comp.cmindex, dist=self.sizedist)
        self.ext.wavel   = wavel
        self.ext.energy  = E_keV
        self.ext.tau_sca = kappa * self.md
        return

    def calc_tau_abs(self, wavel):
        """
        Calculate total absorption cross section for dust population
            | **INPUTS**
            | wavel : Wavelengths to use [Angstroms]
            |
            | **RETURNS**
            | \tau_{abs} = \tau_{ext} - \tau_{sca}
        """
        # If an extinction calculation has already occured, check to see if
        #  it's the same grid and print a warning if it is not.
        if self.ext.wavel is not None:
            _test_wavel_grid(wavel, self)

        # Check if the extinction and scattering cross sections have been calculated
        # If not, run the calculation
        if self.ext.tau_ext is None:
            print("Calculating extinction cross-sections")
            self.calc_tau_ext(wavel)
        if self.ext.tau_sca is None:
            print("Calculating scattering cross-sections")
            self.calc_tau_sca(wavel)

        self.ext.tau_abs = self.ext.tau_ext - self.ext.tau_sca
        return

#---------------------------------------------------------
# Supporting internal functions

def _test_wavel_grid(wavel, gpop):
    if gpop.ext.wavel is not None:
        print("WARNING: Using wavelength grid from previous extinction calculation")
    return

def _pick_scatmodel(sname):
    result = dict(zip(['RG','Mie'], [RGscat(), Mie()]))
    try:
        return result[sname]
    except:
        print("Scattering model not found in library")
        return

#---------------------------------------------------------
# Supporting external functions for creating a GrainPop quickly

def make_MRN_GrainPop(amin=0.005, amax=0.25, p=3.5,
                      compname='Drude', scatname='RG', md=DEFAULT_MD, **kwargs):
    """
    Quick start for making a GrainPop that follows a power law dust grain size distribution
        |
        | **INPUTS**
        | amin : minimum grain size cut off [microns]
        | amax : maximum grain size cut off [microns]
        |
        | **KEWORDS**
        | p : powerlaw index, 3.5 (default)
        | compname : a string describing the composition name : 'Drude' (default), 'Graphite', 'Silicate'
        | scatname : a string describing the scattering model : 'RG' (default), 'Mie'
        | md : dust mass column [g cm^-2], 1.e-4 (default)
    """
    gdist = Powerlaw(amin=amin, amax=amax, p=p, **kwargs)
    comp  = Composition(compname)
    scatm = _pick_scatmodel(scatname)
    return GrainPop(gdist, comp, scatm, md=md)
