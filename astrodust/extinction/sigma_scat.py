
import numpy as np
from scipy.interpolate import interp1d

from .. import constants as c
from . import scatmodels as sms
from ..distlib.composition import cmindex as cmi
from .. import distlib

__all__ = ['ScatModel','DiffScat','SigmaExt','SigmaScat','kappa_ext','kappa_sca']

DEFAULT_MD = 1.e-4  # g cm^-2

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

class ScatModel(object):
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

#-------------- Quickly make a common ScatModel object ---------------------------

def makeScatModel( model_name, material_name ):
    """
    | **INPUTS**
    | model_name    : string : 'RG' or 'Mie'
    | material_name : string : 'Drude', 'Silicate', 'Graphite', 'SmallGraphite'
    |
    | **RETURNS**
    | ScatModel object
    """

    if model_name == 'RG':
        sm = sms.RGscat()
    elif model_name == 'Mie':
        sm = sms.Mie()
    else:
        print('Error: Model name not recognized')
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
        print('Error: CM name not recognized')
        return

    return ScatModel(sm, cm)


#-------------- Various Types of Scattering Cross-sections -----------------------

class DiffScat(object):
    """
    | A differential scattering cross-section [cm^2 ster^-1] integrated
    | over dust grain size distribution
    |
    | **ATTRIBUTES**
    | scatm : ScatModel
    | theta : np.array : arcsec
    | E     : scalar or np.array : Note, must match number of theta values if size > 1
    | a     : scalar : um
    | dsig  : np.array : cm^2 ster^-1
    """
    def __init__(self, scatm=ScatModel(), theta=angles(), E=1.0, a=1.0):
        self.scatm  = scatm
        self.theta  = theta
        self.E      = E
        self.a      = a

        cm   = scatm.cmodel
        scat = scatm.smodel
        # Do not print citation here, because this function is called multiple times by other modules

        if cm.cmtype == 'Graphite':
            dsig_pe = scat.Diff(theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
            dsig_pa = scat.Diff(theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
            self.dsig = (dsig_pa + 2.0 * dsig_pe) / 3.0
        else:
            self.dsig   = scat.Diff(theta=theta, a=a, E=E, cm=cm)

class SigmaScat(object):
    """
    | Scattering cross-section [cm^2] for a single grain size
    |
    | **ATTRIBUTES**
    | scatm : ScatModel
    | E     : scalar or np.array : keV
    | a     : scalar : um
    | qsca  : scalar or np.array : unitless scattering efficiency
    | sigma : scalar or np.array : cm^2
    """
    def __init__(self, scatm=ScatModel(), E=1.0, a=1.0):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        cm   = scatm.cmodel
        scat = scatm.smodel
        print(cm.citation)

        cgeo  = np.pi * np.power(a*c.micron2cm, 2)

        if cm.cmtype == 'Graphite':
            qsca_pe = scat.Qsca(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
            qsca_pa = scat.Qsca(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
            self.qsca = (qsca_pa + 2.0*qsca_pe) / 3.0
        else:
            self.qsca = scat.Qsca(a=a, E=E, cm=cm)

        self.sigma = self.qsca * cgeo

class SigmaExt(object):
    """
    | EXTINCTION cross-section [cm^2] for a single grain size
    |
    | **ATTRIBUTES**
    | scatm : ScatModel
    | E     : scalar or np.array : keV
    | a     : scalar : um
    | qext  : scalar or np.array : unitless extinction efficiency
    | sigma : scalar or np.array : cm^2
    """
    def __init__(self, scatm=ScatModel(), E=1.0, a=1.0):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        if scatm.stype == 'RGscat':
            print('Rayleigh-Gans cross-section not currently supported for KappaExt')
            self.qext = None
            self.sigma = None
            return

        cm   = scatm.cmodel
        scat = scatm.smodel
        print(cm.citation)

        cgeo  = np.pi * np.power(a*c.micron2cm, 2)
        if cm.cmtype == 'Graphite':
            qext_pe = scat.Qext(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
            qext_pa = scat.Qext(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
            self.qext = (qext_pa + 2.0*qext_pe) / 3.0
        else:
            self.qext = scat.Qext(a=a, E=E, cm=cm)
        self.sigma = self.qext * cgeo

def kappa_sca(E, scatm=sms.RGscat(), cm=cmi.CmDrude(), dist=distlib.Powerlaw(), md=DEFAULT_MD):
    """
    | Opacity to scattering [g^-1 cm^2] integrated over dust grain size distribution.
    |
    | **ATTRIBUTES**
    | scatm : ScatModel
    | E     : scalar or np.array : keV
    | dist  : distlib.DustSpectrum
    | kappa : scalar or np.array : cm^2 g^-1, typically
    """
    print(cm.citation)
    ndens = dist.ndens(md)

    cgeo = np.pi * np.power(dist.a * c.micron2cm, 2)

    qsca    = np.zeros(shape=(np.size(E),np.size(dist.a)))
    qsca_pe = np.zeros(shape=(np.size(E),np.size(dist.a)))
    qsca_pa = np.zeros(shape=(np.size(E),np.size(dist.a)))

    # Test for graphite case
    if cm.cmtype == 'Graphite':
        cmGraphitePerp = cmi.CmGraphite(size=cm.size, orient='perp')
        cmGraphitePara = cmi.CmGraphite(size=cm.size, orient='para')

        if np.size(dist.a) > 1:
            for i in range(np.size(dist.a)):
                qsca_pe[:,i] = scatm.Qsca(E, a=dist.a[i], cm=cmGraphitePerp)
                qsca_pa[:,i] = scatm.Qsca(E, a=dist.a[i], cm=cmGraphitePara)
        else:
            qsca_pe = scatm.Qsca(E, a=dist.a, cm=cmGraphitePerp)
            qsca_pa = scatm.Qsca(E, a=dist.a, cm=cmGraphitePara)

        qsca    = (qsca_pa + 2.0 * qsca_pe) / 3.0

    else:
        if np.size(dist.a) > 1:
            for i in range(np.size(dist.a)):
                qsca[:,i] = scatm.Qsca(E, a=dist.a[i], cm=cm)
        else:
            qsca = scatm.Qsca(E, a=dist.a, cm=cm)

    if np.size(dist.a) == 1:
        kappa = ndens * qsca * cgeo / md
    else:
        kappa = np.array([])
        for j in range(np.size(E)):
            kappa = np.append(kappa,
                              c.intz(dist.a, ndens * qsca[j,:] * cgeo) / md)

    return kappa


def kappa_ext(E, scatm=sms.RGscat(), cm=cmi.CmDrude(), dist=distlib.Powerlaw(), md=DEFAULT_MD):
    """
    | Opacity to EXTINCTION [g^-1 cm^2] integrated over dust grain size
    | distribution
    |
    | **ATTRIBUTES**
    | scatm : ScatModel
    | E     : scalar or np.array : keV
    | dist  : distlib.DustSpectrum
    | kappa : scalar or np.array : cm^2 g^-1, typically
    """
    # Check if using the RGscat model, which does not do absorption
    if scatm.stype == 'RGscat':
        print 'Rayleigh-Gans cross-section not currently supported for KappaExt'
        kappa = None
        return kappa

    print(cm.citation)
    ndens = dist.ndens(md)

    cgeo = np.pi * np.power(dist.a * c.micron2cm, 2)

    qext    = np.zeros(shape=(np.size(E),np.size(dist.a)))
    qext_pe = np.zeros(shape=(np.size(E),np.size(dist.a)))
    qext_pa = np.zeros(shape=(np.size(E),np.size(dist.a)))

    # Test for graphite case
    if cm.cmtype == 'Graphite':
        cmGraphitePerp = cmi.CmGraphite(size=cm.size, orient='perp')
        cmGraphitePara = cmi.CmGraphite(size=cm.size, orient='para')

        if np.size(dist.a) > 1:
            for i in range(np.size(dist.a)):
                qext_pe[:,i] = scatm.Qext(E, a=dist.a[i], cm=cmGraphitePerp)
                qext_pa[:,i] = scatm.Qext(E, a=dist.a[i], cm=cmGraphitePara)
        else:
            qext_pe = scatm.Qext(E, a=dist.a, cm=cmGraphitePerp)
            qext_pa = scatm.Qext(E, a=dist.a, cm=cmGraphitePara)

        qext    = (qext_pa + 2.0 * qext_pe) / 3.0

    else:
        if np.size(dist.a) > 1:
            for i in range(np.size(dist.a)):
                qext[:,i] = scatm.Qext(E, a=dist.a[i], cm=cm)
        else:
            qext = scatm.Qext(E, a=dist.a, cm=cm)

    if np.size(dist.a) == 1:
        kappa = ndens * qext * cgeo / md
    else:
        kappa = np.array([])
        for j in range(np.size(E)):
            kappa = np.append(kappa,
                              c.intz(dist.a, ndens * qext[j,:] * cgeo) / md)

    return kappa
