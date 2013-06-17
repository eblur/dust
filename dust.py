
import numpy as np
import constants as c

#-----------------------------------------------------

def adist( amin=0.1, amax=1.0, na=100 ):
    """ FUNCTION adist( amin=0.1, amax=1.0, na=100 )
    Returns np.array of grain sizes between amin and amax (microns) """
    da = (amax-amin)/na
    return np.arange( amin, amax+da, da )

#-----------------------------------------------------

class Grain(object):
    """ OBJECT dust.Grain
    ----------------------------------------
    __init__(self, rad=1.0, rho=3.0)
    ----------------------------------------
    a   : scalar [micron]
    rho : scalar grain density [g cm^-3]
    ----------------------------------------
    FUNCTIONS
    ndens ( md : mass density [g cm^-2 or g cm^-3] )
          returns : scalar number density [cm^-3]
    """
    def __init__(self, rad=1.0, rho=3.0):
        self.a   = rad
        self.rho = rho
    def ndens(self, md=1.5e-5 ):
        gvol = 4./3. * np.pi * np.power( self.a*c.micron2cm(), 3 )
        return md / ( gvol*self.rho )

class Dustdist(object):  # power (p), rho, radius (a)
    """ OBJECT dust.Dustdist
    ----------------------------------------
    __init__(self, p=4.0, rho=3.0, rad=adist() )
    ----------------------------------------
    p   : scalar for power law dn/da \propto a^-p
    rho : scalar grain density [g cm^-3] 
    a   : np.array of grain sizes [microns]
    ----------------------------------------
    FUNCTIONS
    dnda ( md : mass density [g cm^-2 or g cm^-3] )
         returns : number density [cm^-3 um^-1]
    """
    def __init__(self, p=3.5, rho=3.0, rad=adist() ):
        self.p   = p
        self.rho = rho
        self.a   = rad

    def dnda(self, md=1.5e-5):
        adep  = np.power( self.a, -self.p )   # um^-p
        dmda  = adep * 4./3. * np.pi * self.rho * np.power( self.a*c.micron2cm(), 3 ) # g um^-p
        const = md / c.intz( self.a, dmda ) # cm^-? um^p-1
        return const * adep # cm^-? um^-1

class Dustspectrum(object):  #radius (a), number density (nd), and mass density (md)
    """ OBJECT dust.Dustspectrum
    ----------------------------------------
    __init__( self, rad=Dustdist(), md=1.5e-5 )
    ----------------------------------------
    rad : Dustdist or Grain
    md  : mass density of dust
    nd  : number density of dust
    """
    def __init__( self, rad=Dustdist(), md=1.5e-5 ):
        self.md  = md
        self.a   = rad.a

        if type( rad ) == Dustdist:
            self.nd = rad.dnda( md=md )
        if type( rad ) == Grain:
            self.nd = rad.ndens( md=md )

#-----------------------------------------------------------------

def make_dust_spectrum( amin=0.1, amax=1.0, na=100, p=4.0, rho=3.0, md=1.5e-5 ):
    """ FUNCTION make_dust_spectrum( amin=0.1, amax=1.0, na=100, p=4.0, rho=3.0, md=1.5e-5 )
    ----------------------------------------
    INPUTS
    amin : [micron]
    amax : [micron]
    na   : int
    p    : scalar for dust power law dn/da \propto a^-p
    rho  : grain density [g cm^-3]
    md   : mass density [g cm^-2 or g cm^-3]
    ----------------------------------------
    RETURNS
    dust.Dustspectrum (object)
    """
    return Dustspectrum( rad=Dustdist( rad=adist(amin=amin, amax=amax, na=na), p=p, rho=rho ), md=md )


