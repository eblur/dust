import numpy as np

from .sizedist import *
from .composition import *
from ..extinction import sigma_scat as ss
from ..extinction import ExtinctionCurve
from .. import constants as c

# Default dust mass column density
DEFAULT_MD = 1.e-4  # g cm^-2

class GrainPop(object):
    """
    A population of dust grains
        |
        | **INITIALIZE**
        | sizedist : grain size dist objects : distlib.Grain, distlib.Powerlaw, distlib.WD01
        | composition : a string describing the composition type : 'Graphite', 'Silicate' or 'Drude'
        | scatmodel : a string describing the scattering physics to use : 'RG' or 'Mie'
        | md : dust mass column : 1.e-4 g cm^-2 (default)
        |
        | **ATTRIBUTES**
        | sizedist : distlib.Grain(), distlib.Powerlaw(), distlib.WD01()
        | composition : distlib.Composition class
        | scatmodel : extinction.sigma_scat.ScatModel
        | md : dust mass column [g cm^-2]
        | ext : extinction.ExtinctionCurve object
    """
    def __init__(self, sizedist, composition, scatmodel, md=DEFAULT_MD):
        self.sizedist    = sizedist
        self.composition = Composition(composition)
        self.scatmodel   = ss.makeScatModel(scatmodel, composition)
        self.md          = md
        self.ext         = ExtinctionCurve()

    @property
    def a(self):
        return self.sizedist.a

    @property
    def rho(self):
        return self.composition.rho

    @property
    def ndens(self):
        return self.sizedist.ndens(self.md)

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
        kappa = ss.kappa_ext(E_keV, scatm=self.scatmodel,
                             dist=self.sizedist, md=self.md)
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
        kappa = ss.kappa_scat(E_keV, scatm=self.scatmodel,
                              dist=self.sizedist, md=self.md)
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

def _test_wavel_grid(wavel, gpop):
    if np.sum(gpop.ext.wavel - wavel) != 0.0:
        print("WARNING: Using wavelength grid from previous extinction calculation")
    return
