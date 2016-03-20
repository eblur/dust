
import numpy as np
import constants as c

## Some default values

MDUST    = 1.5e-5 # g cm^-2 (dust mass column)
RHO_G    = 3.0    # g cm^-3 (average grain material density)

NA       = 100    # default number for grain size dist resolution
PDIST    = 3.5    # default slope for power law distribution
AMICRON  = 1.0    # grain radius (1 micron)

#---------------------------------------------------------------
# This is gratuitous but it's helpful for reading the code
def make_rad(amin, amax, na, log=False):
    """
    Make a grid of dust grain radius values
    
    | **INPUTS**
    | amin : scalar [micron] : Minimum grain size to use
    | amax : scalar [micron] : Maximum grain size to use
    | na   : integer         : Number of points in grid
    
    | **RETURNS**
    | A numpy array of length na
    
    >>> len(make_rad(0.1,1.0,10)) == 10
    """
    if log:
        return np.logspace( np.log10(amin), np.log10(amax), na )
    else:
        return np.linspace(amin, amax, na)

GREY_RAD = make_rad(0.1, 1.0, NA)
MRN_RAD  = make_rad(0.005, 0.3, NA)

#-----------------------------------------------------

# Print all the default values
def print_defaults():
    print("Default values:")
    print("MDUST = %.3e" % (MDUST))
    print("RHO_G = %.3f" % (RHO_G))
    print("NA = %d" % (NA))
    print("PDIST = %.3f" % (PDIST))
    print("AMICRON = %.3f" % (AMICRON))
    print("GREY_RAD (amin, amax): (%.3f, %.3f)" % (GREY_RAD[0], GREY_RAD[-1]))
    print("MRN_RAD (amin, amax): (%.3f, %.3f)" % (MRN_RAD[0], MRN_RAD[-1]))
    return

#-----------------------------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    | rho : scalar grain density [g cm^-3]

    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* scalar number density [cm^-3]
    
    >>> Grain().ndens(0.0) == 0.0
    >>> Grain(rho=0.0).ndens() == np.inf
    """
    def __init__(self, rad=AMICRON, rho=RHO_G):
        self.a   = rad
        self.rho = rho
    def ndens(self, md=MDUST ):
        gvol = 4./3. * np.pi * np.power( self.a*c.micron2cm, 3 )
        return md / ( gvol*self.rho )

class Dustdist(object):
    """ 
    | **ATTRIBUTES**
    | p   : scalar for power law dn/da \propto a^-p
    | rho : scalar grain density [g cm^-3] 
    | a   : np.array of grain sizes [microns]

    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* number density [cm^-3 um^-1]
    
    >>> np.sum(Dustdist().ndens(0.0)) == 0.0
    >>> np.all(np.isinf(Dustdist(rho=0.0).ndens()))
    """
    def __init__(self, amin=MRN_RAD[0], amax=MRN_RAD[-1], p=PDIST, rho=RHO_G, \
                 na=NA, log=False):
        self.amin = amin
        self.amax = amax
        self.a    = make_rad(amin, amax, na, log=log)
        self.p    = p
        self.rho  = rho
    
    def ndens(self, md=MDUST):
        adep  = np.power( self.a, -self.p )   # um^-p
        dmda  = adep * 4./3. * np.pi * self.rho * np.power( self.a*c.micron2cm, 3 ) # g um^-p
        const = md / c.intz( self.a, dmda ) # cm^-? um^p-1
        return const * adep # cm^-? um^-1


class Dustspectrum(object):  #radius (a), number density (nd), and mass density (md)
    """
    | **ATTRIBUTES**
    | dist : A dust distribution that contains attributes a and rho and ndens function
    | md   : mass density of dust [units arbitrary, usually g cm^-2]
    | nd   : number density of dust [set by md units]
    | rho  : dust grain material density [g cm^-3]
    
    >>> np.sum(Dustspectrum(md=0.0).nd) == 0
    >>> np.abs(_integrate_dust_mass(Dustspectrum())/MDUST - 1.0) < 0.01
    >>> np.abs(_integrate_dust_mass(Dustspectrum(rad=Grain()))/MDUST - 1.0) < 0.01
    """
    def __init__( self, rad=Dustdist(), md=MDUST ):
        self.md  = md
        self.a   = rad.a
        self.rho = rad.rho
        self.nd  = rad.ndens(md)

#-----------------------------------------------------------------

def MRN_dist(amin, amax, p, rho=RHO_G, md=MDUST, **kwargs):
    """
    Returns a dust spectrum for a power law distribution of dust grains
    
    | **INPUTS**
    | amin : [micron]
    | amax : [micron]
    | na   : int
    | p    : scalar for dust power law dn/da \propto a^-p
    | rho  : grain density [g cm^-3]
    | md   : mass density [g cm^-2 or g cm^-3]

    | **RETURNS**
    | dust.Dustspectrum( object )
    """
    ddist = Dustdist(amin=amin, amax=amax, p=p, rho=rho, **kwargs)
    return Dustspectrum(rad=ddist, md=md)

def make_dust_spectrum(amin=0.1, amax=1.0, na=100, p=4.0, rho=3.0, md=1.5e-5):
    print("WARNING: make_dust_spectrum is deprecated. Use MRN_dist")
    return MRN_dist(amin, amax, p, na=na, rho=rho, md=md)

##------------------------------------------------------------------------
## Functions for testing purposes

def _integrate_dust_mass(dustspec):
    from scipy.integrate import trapz
    mass = (4.*np.pi/3.) * dustspec.rho * (c.micron2cm*dustspec.a)**3
    if np.size(dustspec.a) == 1:
        return mass * dustspec.nd
    else:
        return trapz(mass * dustspec.nd, dustspec.a)
