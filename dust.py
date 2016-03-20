
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
    """
    if log:
        return np.logspace( np.log10(amin), np.log10(amax), na )
    else:
        return np.linspace(amin, amax, na)

GREY_RAD = make_rad(0.1, 1.0, NA)
MRN_RAD  = make_rad(0.005, 0.3, NA)

#-----------------------------------------------------

class Grain(object):
    """
    | **ATTRIBUTES**
    | a   : scalar [micron]
    | rho : scalar grain density [g cm^-3]

    | **FUNCTIONS**
    | ndens ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* scalar number density [cm^-3]
    """
    def __init__(self, rad=AMICRON, rho=RHO_G):
        self.a   = rad
        self.rho = rho
    def ndens(self, md=MDUST ):
        gvol = 4./3. * np.pi * np.power( self.a*c.micron2cm, 3 )
        return md / ( gvol*self.rho )

class Dustdist(object):  # power (p), rho, radius (a)
    """ 
    | **ATTRIBUTES**
    | p   : scalar for power law dn/da \propto a^-p
    | rho : scalar grain density [g cm^-3] 
    | a   : np.array of grain sizes [microns]

    | **FUNCTIONS**
    | dnda ( md : mass density [g cm^-2 or g cm^-3] )
    |    *returns* number density [cm^-3 um^-1]
    """
    def __init__(self, rad=MRN_RAD, p=PDIST, rho=RHO_G):
        self.p   = p
        self.rho = rho
        self.a   = rad

    def dnda(self, md=MDUST):
        adep  = np.power( self.a, -self.p )   # um^-p
        dmda  = adep * 4./3. * np.pi * self.rho * np.power( self.a*c.micron2cm, 3 ) # g um^-p
        const = md / c.intz( self.a, dmda ) # cm^-? um^p-1
        return const * adep # cm^-? um^-1

class Dustspectrum(object):  #radius (a), number density (nd), and mass density (md)
    """
    | **ATTRIBUTES**
    | rad : Dustdist or Grain
    | md  : mass density of dust [units arbitrary, usually g cm^-2]
    | nd  : number density of dust [set by md units]
    """
    def __init__( self, rad=Dustdist(), md=MDUST ):
        self.md  = md
        self.a   = rad.a

        if type( rad ) == Dustdist:
            self.nd = rad.dnda( md=md )
        if type( rad ) == Grain:
            self.nd = rad.ndens( md=md )

#-----------------------------------------------------------------

def MRN_dist(amin, amax, p, na=NA, rho=RHO_G, md=MDUST, log=False):
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
    radii = make_rad(amin, amax, na, log=log)
    ddist = Dustdist(rad=radii, p=p, rho=rho)
    return Dustspectrum(rad=ddist, md=md)

def make_dust_spectrum(amin=0.1, amax=1.0, na=100, p=4.0, rho=3.0, md=1.5e-5):
    print("WARNING: make_dust_spectrum is deprecated. Use MRN_dist")
    return MRN_dist(amin, amax, p, na=na, rho=rho, md=md)


