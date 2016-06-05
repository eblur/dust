
import numpy as np
from .. import constants as c

__all__ = ['Grain', 'Powerlaw']

# Some default values
MDUST    = 1.5e-5  # g cm^-2 (dust mass column)
RHO      = 3.0     # g cm^-3 (average grain material density)

NA       = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution
AMICRON  = 1.0     # grain radius (1 micron)

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron

# ---------------------------------------------------------------
# This is gratuitous but it's helpful for reading the code

def make_rad(amin, amax, na, log=False):
    """
    | Make a grid of dust grain radius values
    |
    | **INPUTS**
    | amin : scalar [micron] : Minimum grain size to use
    | amax : scalar [micron] : Maximum grain size to use
    | na   : integer         : Number of points in grid
    |
    | **RETURNS**
    | A numpy array of length na

    >>> len(make_rad(0.1,1.0,10)) == 10
    """
    if log:
        return np.logspace(np.log10(amin), np.log10(amax), na)
    else:
        return np.linspace(amin, amax, na)

GREY_RAD = make_rad(0.1, 1.0, NA)
MRN_RAD  = make_rad(AMIN, AMAX, NA)

# -----------------------------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    |
    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* scalar number density [cm^-3]
    """
    def __init__(self, rad=AMICRON, rho=RHO):
        assert np.size(rad) == 1
        self.a   = np.array([rad])

    def ndens(self, md, rho=RHO):
        gvol = 4. / 3. * np.pi * np.power(self.a*c.micron2cm, 3)
        return md / (gvol * rho)

class Powerlaw(object):
    """
    | **ATTRIBUTES**
    | p   : scalar for power law dn/da \propto a^-p
    | rho : scalar grain density [g cm^-3]
    | a   : np.array of grain sizes [microns]

    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* number density [cm^-3 um^-1]

    >>> np.sum(Powerlaw().ndens(0.0)) == 0.0
    >>> np.all(np.isinf(Powerlaw(rho=0.0).ndens()))
    """
    def __init__(self, amin=MRN_RAD[0], amax=MRN_RAD[-1], p=PDIST,
                 na=NA, log=False):
        self.amin = amin
        self.amax = amax
        self.a    = make_rad(amin, amax, na, log=log)
        self.p    = p

    def ndens(self, md, rho=RHO):
        adep  = np.power(self.a, -self.p)   # um^-p
        gdens = (4. / 3.) * np.pi * rho
        dmda  = adep * gdens * np.power(self.a * c.micron2cm, 3)  # g um^-p
        const = md / c.intz(self.a, dmda)  # cm^-? um^p-1
        return const * adep  # cm^-? um^-1

# For backwards compatibility
def Dustdist(amin=MRN_RAD[0], amax=MRN_RAD[-1], p=PDIST, rho=RHO, na=NA, log=False):
    print("WARNING: dust.Dustdist is deprecated. Use Powerlaw")
    return Powerlaw(amin=amin, amax=amax, p=p, rho=rho, na=na, log=log)

'''
# 2016.06.05 - deprecated
class DustSpectrum(object):  # radius (a), number density (nd), and mass density (md)
    """
    | **ATTRIBUTES**
    | dist : A dust distribution that contains attributes a and rho and ndens function
    | md   : mass density of dust [units arbitrary, usually g cm^-2]
    | nd   : number density of dust [set by md units]
    | rho  : dust grain material density [g cm^-3]

    >>> np.sum(DustSpectrum(md=0.0).nd) == 0

    >>> test_grain = DustSpectrum()
    >>> test_grain.calc_from_dist(Grain())
    >>> np.abs(_integrate_dust_mass(test_grain)/MDUST - 1.0) < 0.01

    >>> test_powlaw = DustSpectrum()
    >>> test_powlaw.calc_from_dist(Powerlaw())
    >>> np.abs(_integrate_dust_mass(test_powlaw)/MDUST - 1.0) < 0.01
    """
    def __init__(self, a=None, rho=None, nd=None, md=None):
        self.md  = md
        self.a   = a
        self.rho = rho
        self.nd  = nd

    def calc_from_dist(self, dist, md=MDUST):
        self.md  = md
        self.a   = dist.a
        self.rho = dist.rho
        self.nd  = dist.ndens(md)

    def integrate_dust_mass(self):
        from scipy.integrate import trapz
        mass = (4.*np.pi/3.) * self.rho * (c.micron2cm*self.a)**3
        if np.size(self.a) == 1:
            return mass * self.nd
        else:
            return trapz(mass * self.nd, self.a)
'''

#-----------------------------------------------------------------

'''
# 2016.06.05 - deprecated
def MRN_dist(amin=AMIN, amax=AMAX, p=PDIST, md=MDUST, **kwargs):
    """
    | Returns a dust spectrum for a power law distribution of dust grains
    |
    | **INPUTS**
    | amin : [micron]
    | amax : [micron]
    | p    : scalar for dust power law dn/da \propto a^-p
    | rho  : grain density [g cm^-3]
    | md   : mass density [g cm^-2 or g cm^-3]
    | **kwargs : See distlib.sizedist.Powerlaw keywords
    |
    | **RETURNS**
    | dust.DustSpectrum object
    """
    ddist  = Powerlaw(amin=amin, amax=amax, p=p, rho=rho, **kwargs)
    result = DustSpectrum()
    result.calc_from_dist(ddist, md=md)
    return result
'''
# Depreceated
#def make_dust_spectrum(amin=0.1, amax=1.0, na=100, p=4.0, rho=3.0, md=1.5e-5):
#    print("WARNING: make_dust_spectrum is deprecated. Use MRN_dist")
#    return MRN_dist(amin, amax, p, na=na, rho=rho, md=md)
