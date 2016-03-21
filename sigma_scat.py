
import numpy as np
#np.seterr(all='warn')
import constants as c
import dust
import cmindex as cmi
import scatmodels as sms
from scipy.interpolate import interp1d

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
    | **ATTRIBUTES**
    | smodel : scattering model object : RGscat(), Mie()
    | cmodel : cmindex object : CmDrude(), CmGraphite(), CmSilicate()
    | stype  : string : 'RGscat', 'Mie'
    | cmtype : 'Drude', 'Silicate', 'Graphite'
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
    | **INPUTS**
    | model_name    : string : 'RG' or 'Mie'
    | material_name : string : 'Drude', 'Silicate', 'Graphite', 'SmallGraphite'

    | **RETURNS**
    | Scatmodel object
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
    A differential scattering cross-section [cm^2 ster^-1] integrated
    over dust grain size distribution

    | **ATTRIBUTES**
    | scatm : Scatmodel
    | theta : np.array : arcsec
    | E     : scalar or np.array : Note, must match number of theta values if size > 1
    | a     : scalar : um
    | dsig  : np.array : cm^2 ster^-1
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
    Total scattering cross-section [cm^2] integrated over a dust grain
    size distribution

    | **ATTRIBUTES**
    | scatm : Scatmodel
    | E     : scalar or np.array : keV
    | a     : scalar : um
    | qsca  : scalar or np.array : unitless scattering efficiency
    | sigma : scalar or np.array : cm^2
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
    Total EXTINCTION cross-section [cm^2] integrated over a dust grain
    size distribution

    | **ATTRIBUTES**
    | scatm : Scatmodel
    | E     : scalar or np.array : keV
    | a     : scalar : um
    | qext  : scalar or np.array : unitless extinction efficiency
    | sigma : scalar or np.array : cm^2
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
    Opacity to scattering [g^-1 cm^2] integrated over dust grain size distribution.
    
    | **ATTRIBUTES**
    | scatm : Scatmodel
    | E     : scalar or np.array : keV
    | dist  : dust.Dustspectrum
    | kappa : scalar or np.array : cm^2 g^-1, typically
    """
    def __init__( self, E=1.0, scatm=Scatmodel(), dist=dust.Dustspectrum() ):
        self.scatm  = scatm
        self.E      = E
        self.dist   = dist

        cm   = scatm.cmodel
        scat = scatm.smodel

        cgeo = np.pi * np.power( dist.a * c.micron2cm(), 2 )

        qsca    = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
        qsca_pe = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
        qsca_pa = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
                                
        # Test for graphite case
        if cm.cmtype == 'Graphite':
            cmGraphitePerp = cmi.CmGraphite(size=cm.size, orient='perp')
            cmGraphitePara = cmi.CmGraphite(size=cm.size, orient='para')

            if np.size(dist.a) > 1:
                for i in range( np.size(dist.a) ):
                    qsca_pe[:,i] = scat.Qsca( E, a=dist.a[i], cm=cmGraphitePerp )
                    qsca_pa[:,i] = scat.Qsca( E, a=dist.a[i], cm=cmGraphitePara )
            else:
                qsca_pe = scat.Qsca( E, a=dist.a, cm=cmGraphitePerp )
                qsca_pa = scat.Qsca( E, a=dist.a, cm=cmGraphitePara )
            
            qsca    = ( qsca_pa + 2.0 * qsca_pe ) / 3.0

        else:
            if np.size(dist.a) > 1:
                for i in range( np.size(dist.a) ):
                    qsca[:,i] = scat.Qsca( E, a=dist.a[i], cm=cm )
            else:
                qsca = scat.Qsca( E, a=dist.a, cm=cm )

        if np.size(dist.a) == 1:
            kappa = dist.nd * qsca * cgeo / dist.md
        else:
            kappa = np.array([])
            for j in range( np.size(E) ):
                kappa = np.append( kappa, \
                                   c.intz( dist.a, dist.nd * qsca[j,:] * cgeo ) / dist.md )

        self.kappa = kappa


class Kappaext(object):
    """
    Opacity to EXTINCTION [g^-1 cm^2] integrated over dust grain size
    distribution
    
    | **ATTRIBUTES**
    | scatm : Scatmodel
    | E     : scalar or np.array : keV
    | dist  : dust.Dustspectrum
    | kappa : scalar or np.array : cm^2 g^-1, typically
    """
    def __init__( self, E=1.0, scatm=Scatmodel(), dist=dust.Dustspectrum() ):
        self.scatm  = scatm
        self.E      = E
        self.dist   = dist

        if scatm.stype == 'RG':
            print 'Rayleigh-Gans cross-section not currently supported for Kappaext'
            self.kappa = None
            return

        cm   = scatm.cmodel
        scat = scatm.smodel

        cgeo = np.pi * np.power( dist.a * c.micron2cm(), 2 )

        qext    = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
        qext_pe = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
        qext_pa = np.zeros( shape=( np.size(E),np.size(dist.a) )  )
                                
        # Test for graphite case
        if cm.cmtype == 'Graphite':
            cmGraphitePerp = cmi.CmGraphite(size=cm.size, orient='perp')
            cmGraphitePara = cmi.CmGraphite(size=cm.size, orient='para')

            if np.size(dist.a) > 1:
                for i in range( np.size(dist.a) ):
                    qext_pe[:,i] = scat.Qext( E, a=dist.a[i], cm=cmGraphitePerp )
                    qext_pa[:,i] = scat.Qext( E, a=dist.a[i], cm=cmGraphitePara )
            else:
                qext_pe = scat.Qext( E, a=dist.a, cm=cmGraphitePerp )
                qext_pa = scat.Qext( E, a=dist.a, cm=cmGraphitePara )
            
            qext    = ( qext_pa + 2.0 * qext_pe ) / 3.0

        else:
            if np.size(dist.a) > 1:
                for i in range( np.size(dist.a) ):
                    qext[:,i] = scat.Qext( E, a=dist.a[i], cm=cm )
            else:
                qext = scat.Qext( E, a=dist.a, cm=cm )

        if np.size(dist.a) == 1:
            kappa = dist.nd * qext * cgeo / dist.md
        else:
            kappa = np.array([])
            for j in range( np.size(E) ):
                kappa = np.append( kappa, \
                                   c.intz( dist.a, dist.nd * qext[j,:] * cgeo ) / dist.md )

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
