
import numpy as np
from .. import constants as c
from scipy.integrate import trapz

__all__ = ['Grain', 'Powerlaw']

# Some default values
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution
AMICRON  = 1.0     # grain radius (1 micron)

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron

#-----------------------------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    |
    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* scalar number density [cm^-3]
    """
    def __init__(self, rad=AMICRON):
        assert np.size(rad) == 1
        self.a   = np.array([rad])

    def ndens(self, md, rho):
        """
        Calculate number density of dust grains
            |
            | **INPUTS**
            | md : dust mass density [e.g. g cm^-2]
            | rho : grain material density [g cm^-3]
        """
        gvol = 4. / 3. * np.pi * np.power(self.a*c.micron2cm, 3)
        return md / (gvol * rho)

class Powerlaw(object):
    """
    | **ATTRIBUTES**
    | amin : minimum grain size [microns]
    | amax : maximum grain size [microns]
    | p   : scalar for power law dn/da \propto a^-p
    | NA  : int : number of a values to use
    | log : boolean : False (default), True = use log-spaced a values
    """
    def __init__(self, amin=AMIN, amax=AMAX, p=PDIST, na=NA, log=False):
        self.amin = amin
        self.amax = amax
        if log:
            self.a = np.logspace(amin, amax, na)
        else:
            self.a = np.linspace(amin, amax, na)
        self.p    = p

    def ndens(self, md, rho=RHO):
        """
        Calculate number density of dust grains as a function of grain size
            | **RETURNS** numpy.ndarray of dn/da values [number density per micron]
            |
            | **INPUTS**
            | md : dust mass density [e.g. g cm^-2]
            | rho : grain material density [g cm^-3]
        """
        adep  = np.power(self.a, -self.p)   # um^-p
        gdens = (4. / 3.) * np.pi * rho
        dmda  = adep * gdens * np.power(self.a * c.micron2cm, 3)  # g um^-p
        const = md / trapz(dmda, self.a)  # cm^-? um^p-1
        return const * adep  # cm^-? um^-1
