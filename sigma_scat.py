
import numpy as np
#np.seterr(all='warn')
import constants as c
import dust
import cmindex as cmi
import scatmodels as sms
from scipy.interpolate import interp1d

# from multiprocessing import Pool
#from pathos.multiprocessing import Pool

#----------------------------------------------------------
# evals( emin=1.0, emax=2.0, de=0.1 ) : np.array [keV]
# angles( thmin=5.0, thmax=100.0, dth=5.0 ) : np.array [arcsec]
#

def evals( emin=1.0, emax=2.0, de=0.1 ):
    """ 
    FUNCTION evals( emin=1.0, emax=2.0, de=0.1 )
    RETURNS : np.array
    Distribution of energies [keV]
    """
    return np.arange( emin, emax+de, de )

def angles( thmin=5.0, thmax=100.0, dth=5.0 ):
    """
    FUNCTION angles( thmin=5.0, thmax=100.0, dth=5.0 )
    RETURNS : np.array
    Distribution of angles [arcsec]
    """
    return np.arange( thmin, thmax+dth, dth )

#-------------- Tie scattering mechanism to an index of refraction ------------------

class Scatmodel(object):
    """
    OBJECT Scatmodel( smodel=RGscat(), cmodel=cmi.CmDrude() )
    smodel : scattering model type object : RGscat(), Mie()
    cmodel : cmindex type object : CmDrude(), CmGraphite(), CmSilicate()
    stype  : string : 'RGscat', 'Mie'
    cmtype : 'Drude', 'Silicate', 'Graphite'
    """
    def __init__( self, smodel=sms.RGscat(), cmodel=cmi.CmDrude() ):
        self.smodel = smodel
        self.cmodel = cmodel
        self.stype  = smodel.stype
        self.cmtype = cmodel.cmtype
        # cmtype choices : 'Drude' (requires rho term only)
        #                  'Graphite' (Carbonaceous grains)
        #                  'Silicate' (Astrosilicate)
        #                  --- Graphite and Silicate values come from Draine (2003)

#-------------- Quickly make a common Scatmodel object ---------------------------

def makeScatmodel( model_name, material_name ):
    """
    FUNCTION makeScatmodel( model_name, material_name )
    RETURNS Scatmodel object
    ----------------------------------------------------
    model_name    : string : 'RG' or 'Mie'
    material_name : string : 'Drude', 'Silicate', 'Graphite', 'SmallGraphite'
    """

    if model_name == 'RG':
        sm = sms.RGscat()
    elif model_name == 'Mie':
        sm = sms.Mie()
    else:
        print 'Error: Model name not recognized'
        return

    if material_name == 'Drude':
        cm = cmi.CmDrude()
    elif material_name == 'Silicate':
        cm = cmi.CmSilicate()
    elif material_name == 'Graphite':
        cm = cmi.CmGraphite()
    elif material_name == 'SmallGraphite': # Small Graphite ~ 0.01 um
        cm = cmi.CmGraphite( size='small' )
    else:
        print 'Error: CM name not recognized'
        return

    return Scatmodel( sm, cm )


#-------------- Various Types of Scattering Cross-sections -----------------------

class Diffscat(object):
    """
    A differential scattering cross-section [cm^2 ster^-1]
    --------------------------------------------------------------
    OBJECT Diffscat( scatm=Scatmodel(), theta=angles() [arcsec], E=1.0 [keV], a=1.0 [um] )
    scatm : Scatmodel
    theta : np.array : arcsec
    E     : scalar or np.array : Note, must match number of theta values if size > 1
    a     : scalar : um
    dsig  : np.array : cm^2 ster^-1
    """
    def __init__( self, scatm=Scatmodel(), theta=angles(), E=1.0, a=1.0 ):
        self.scatm  = scatm
        self.theta  = theta
        self.E      = E
        self.a      = a

        cm   = scatm.cmodel
        scat = scatm.smodel

        if cm.cmtype == 'Graphite':
            dsig_pe = scat.Diff( theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp') )
            dsig_pa = scat.Diff( theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para') )
            self.dsig = ( dsig_pa + 2.0 * dsig_pe ) / 3.0
        else:
            self.dsig   = scat.Diff( theta=theta, a=a, E=E, cm=cm )

class Sigmascat(object):
    """
    Total scattering cross-section [cm^2]
    ---------------------------------------------------------
    OBJECT Sigmascat( scatm=Scatmodel(), E=1.0 [keV], a=1.0 [um] )
    scatm : Scatmodel
    E     : scalar or np.array : keV
    a     : scalar : um
    qsca  : scalar or np.array : unitless scattering efficiency
    sigma : scalar or np.array : cm^2
    """
    def __init__( self, scatm=Scatmodel(), E=1.0, a=1.0 ):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        cm   = scatm.cmodel
        scat = scatm.smodel

        cgeo  = np.pi * np.power( a*c.micron2cm(), 2 )

        if cm.cmtype == 'Graphite':
            qsca_pe = scat.Qsca( a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp') )
            qsca_pa = scat.Qsca( a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para') )
            self.qsca = ( qsca_pa + 2.0*qsca_pe ) / 3.0
        else:
            self.qsca = scat.Qsca( a=a, E=E, cm=cm )

        self.sigma = self.qsca * cgeo

class Sigmaext(object):
    """
    Total EXTINCTION cross-section [cm^2]
    ---------------------------------------------------------
    OBJECT Sigmascat( scatm=Scatmodel(), E=1.0 [keV], a=1.0 [um] )
    scatm : Scatmodel
    E     : scalar or np.array : keV
    a     : scalar : um
    qext  : scalar or np.array : unitless extinction efficiency
    sigma : scalar or np.array : cm^2
    """
    def __init__( self, scatm=Scatmodel(), E=1.0, a=1.0 ):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        if scatm.stype == 'RG':
            print 'Rayleigh-Gans cross-section not currently supported for Kappaext'
            self.sigma = None
            return

        cm   = scatm.cmodel
        scat = scatm.smodel

        cgeo  = np.pi * np.power( a*c.micron2cm(), 2 )

        if cm.cmtype == 'Graphite':
            qext_pe = scat.Qext( a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp') )
            qext_pa = scat.Qext( a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para') )
            self.qext = ( qext_pa + 2.0*qext_pe ) / 3.0
        else:
            self.qext = scat.Qext( a=a, E=E, cm=cm )

        self.sigma = self.qext * cgeo

class Kappascat(object):
    """
    Opacity to scattering [g^-1 cm^2]
    OBJECT Kappascat( E=1.0 [keV], scatm=Scatmodel(), dist=dust.Dustspectrum() )
    ---------------------------------
    E     : scalar or np.array : keV, photon energy
    scatm : Scatmodel
    dist  : dust.Dustspectrum
    kappa : scalar or np.array : cm^2 g^-1, typically
    
    ---------------------------------
    To call this class:
    ---------------------------------
    kappa = Kappascat()
    kappa(E) computes integral of scattering cross-section over grain size distribution    
    """
    def __init__( self, E=1.0, scatm=Scatmodel(), dist=dust.Dustspectrum() ):
        self.E      = E
        self.kappa  = None
        self.scatm  = scatm
        self.dist   = dist

    def __call__(self, with_mp=False):
        cm   = self.scatm.cmodel
        scat = self.scatm.smodel

        cgeo = np.pi * np.power( self.dist.a * c.micron2cm(), 2 )
        qsca = np.zeros( shape=( np.size(self.E),np.size(self.dist.a) )  )

        # Test for graphite case
        if cm.cmtype == 'Graphite':
            qsca_pe = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
            qsca_pa = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
            if np.size(self.dist.a) > 1:
                for i in range( np.size(self.dist.a) ):
                    qsca_pe[:,i] = scat.Qsca( self.E, a=self.dist.a[i], cm=cmi.CmGraphite(size=cm.size, orient='perp') )
                    qsca_pa[:,i] = scat.Qsca( self.E, a=self.dist.a[i], cm=cmi.CmGraphite(size=cm.size, orient='para') )
            else:
                qsca_pe = scat.Qsca( self.E, a=self.dist.a, cm=cmi.CmGraphite(size=cm.size, orient='perp') )
                qsca_pa = scat.Qsca( self.E, a=self.dist.a, cm=cmi.CmGraphite(size=cm.size, orient='para') )
            
            qsca = ( qsca_pa + 2.0 * qsca_pe ) / 3.0

        else:
            if np.size(self.dist.a) > 1:
                if with_mp:
                    pool = Pool(processes=2)
                    qsca = np.array(pool.map(self._one_scatter,self.dist.a)).T
                else:
                    for i in range( np.size(self.dist.a) ):
                        qsca[:,i] = self._one_scatter(self.dist.a[i])
            else:
                qsca = scat.Qsca( self.E, a=self.dist.a, cm=cm )

        if np.size(self.dist.a) == 1:
            kappa = self.dist.nd * qsca * cgeo / self.dist.md
        else:
            kappa = np.array([])
            for j in range( np.size(self.E) ):
                kappa = np.append( kappa, \
                                   c.intz( self.dist.a, self.dist.nd * qsca[j,:] * cgeo ) / self.dist.md )

        self.kappa = kappa

    def _one_scatter(self, a):
        """Do one scattering calculation."""
        return self.scatm.smodel.Qsca( self.E, a=a, cm=self.scatm.cmodel )


class Kappaext(object):
    """
    Opacity to EXTINCTION [g^-1 cm^2]
    OBJECT Kappaext( E=1.0 [keV], scatm=Scatmodel(), dist=dust.Dustspectrum() )
    ---------------------------------
    scatm : Scatmodel
    E     : scalar or np.array : keV
    dist  : dust.Dustspectrum
    kappa : scalar or np.array : cm^2 g^-1, typically
        
    ---------------------------------
    To call this class:
    ---------------------------------
    kappa = Kappaext()
    kappa(E) computes integral of extinction cross-section over grain size distribution
    """
    def __init__( self, E=1.0, scatm=Scatmodel(), dist=dust.Dustspectrum() ):
        self.scatm  = scatm
        self.E      = E
        self.dist   = dist

        if scatm.stype == 'RG':
            print 'Rayleigh-Gans cross-section not currently supported for Kappaext'
            self.kappa = None
            return

    def __call__(self):
        cm   = self.scatm.cmodel
        scat = self.scatm.smodel

        cgeo = np.pi * np.power( self.dist.a * c.micron2cm(), 2 )
        qext    = np.zeros( shape=( np.size(self.E),np.size(self.dist.a) )  )
                                
        # Test for graphite case
        if cm.cmtype == 'Graphite':

            qext_pe = np.zeros( shape=( np.size(self.E),np.size(self.dist.a) )  )
            qext_pa = np.zeros( shape=( np.size(self.E),np.size(self.dist.a) )  )

            if np.size(dist.a) > 1:
                for i in range( np.size(dist.a) ):
                    qext_pe[:,i] = scat.Qext( self.E, a=self.dist.a[i], cm=cmi.CmGraphite(size=cm.size, orient='perp') )
                    qext_pa[:,i] = scat.Qext( self.E, a=self.dist.a[i], cm=cmi.CmGraphite(size=cm.size, orient='para') )
            else:
                qext_pe = scat.Qext( self.E, a=self.dist.a, cm=cmi.CmGraphite(size=cm.size, orient='perp') )
                qext_pa = scat.Qext( self.E, a=self.dist.a, cm=cmi.CmGraphite(size=cm.size, orient='para') )
            
            qext    = ( qext_pa + 2.0 * qext_pe ) / 3.0

        else:
            if np.size(self.dist.a) > 1:
                for i in range( np.size(self.dist.a) ):
                    qext[:,i] = scat.Qext( self.E, a=self.dist.a[i], cm=cm )
            else:
                qext = scat.Qext( self.E, a=self.dist.a, cm=cm )

        if np.size(self.dist.a) == 1:
            kappa = self.dist.nd * qext * cgeo / self.dist.md
        else:
            kappa = np.array([])
            for j in range( np.size(self.E) ):
                kappa = np.append( kappa, \
                                   c.intz( self.dist.a, self.dist.nd * qext[j,:] * cgeo ) / self.dist.md )

        self.kappa = kappa


                                                        

#-------------- Objects that can be used for interpolation later -----------------

class KappaSpec( object ):
    """
    OBJECT Kappaspec( E=None, kappa=None, scatm=None, dspec=None )
    E     : np.array : keV
    scatm : Scatmodel
    dspec : dust.Dustspectrum
    kappa : scipy.interpolate.interp1d object with (E, kappa) as arguments
    """
    def __init__(self, E=None, kappa=None, scatm=None, dspec=None ):
        self.E = E
        self.kappa = interp1d( E, kappa )
        self.scatm = scatm
        self.dspec = dspec

class SigmaSpec( object ):
    """
    OBJECT Sigmaspec( E=None, sigma=None, scatm=None )
    E     : np.array : keV
    scatm : Scatmodel
    sigma : scipy.interpolate.interp1d object with (E, sigma) as arguments
    """
    def __init__(self, E=None, sigma=None, scatm=None):
        self.E = E
        self.sigma = interp1d( E, sigma )
        self.scatm = scatm
