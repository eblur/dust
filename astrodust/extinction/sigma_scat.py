
import numpy as np

from .. import constants as c
from . import scatmodels as sms
from ..distlib.composition import cmindex as cmi
from ..distlib import *

__all__ = ['diff_scat','sigma_sca','sigma_ext','kappa_ext','kappa_sca']

DEFAULT_MD = 1.e-4  # g cm^-2
DEFAULT_ALPHA = np.linspace(5.0, 100.0, 20)

#-------------- Various Types of Scattering Cross-sections -----------------------

def diff_scat(a, E, scatm=sms.RGscat(), cm=cmi.CmDrude(), theta=DEFAULT_ALPHA):
    """
    Calculate differential scattering cross-section [cm^2 ster^-1] for a single grain size
        |
        | **INPUTS**
        | a     : scalar : um
        | E     : scalar or np.array : Note, must match number of theta values if size > 1
        |
        | **KEYWORDS**
        | scatm : scatmodel object : scatmodels.RGscat (default)
        | cm    : complex index of refraction object : distlib.composition.cmindex.CmDrude (default)
        | theta : np.array : scattering angle in arcsec, np.linspace(5, 100, 20) (default)
    """
    # Do not print citation here, because this function is called multiple times by other modules
    if cm.cmtype == 'Graphite':
        dsig_pe = scatm.Diff(theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
        dsig_pa = scatm.Diff(theta=theta, a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
        dsig = (dsig_pa + 2.0 * dsig_pe) / 3.0
    else:
        dsig   = scatm.Diff(theta=theta, a=a, E=E, cm=cm)
    return dsig

def sigma_sca(a, E, scatm=sms.RGscat(), cm=cmi.CmDrude(), qval=False):
    """
    Returns scattering cross-section [cm^2] for a single grain size
        |
        | **INPUTS**
        | a     : scalar : um
        | E     : np.ndarray : keV
        |
        | **KEYWORDS**
        | scatm : scatmodel object : extinction.RGscat (default)
        | cm    : complex index of refraction object : distlib.composition.cmindex.CmDrude (default)
        | qval  : boolean : False (default), set True to return scattering efficiency, Qscat [unitless]
    """
    print(cm.citation)

    if cm.cmtype == 'Graphite':
        qsca_pe = scatm.Qsca(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
        qsca_pa = scatm.Qsca(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
        qsca = (qsca_pa + 2.0*qsca_pe) / 3.0
    else:
        qsca = scatm.Qsca(a=a, E=E, cm=cm)

    if qval:
        return qsca
    else:
        cgeo  = np.pi * np.power(a*c.micron2cm, 2)
        return qsca * cgeo

def sigma_ext(a, E, scatm=sms.RGscat(), cm=cmi.CmDrude(), qval=False):
    """
    Returns extinction cross-section [cm^2] for a single grain size
        |
        | **INPUTS**
        | a     : scalar : um
        | E     : np.ndarray : keV
        |
        | **KEYWORDS**
        | scatm : scatmodel object : extinction.RGscat (default)
        | cm    : complex index of refraction object : distlib.composition.cmindex.CmDrude (default)
        | qval  : boolean : False (default), set True to return extinction efficiency, Qext [unitless]
    """
    if scatm.stype == 'RGscat':
        print('Rayleigh-Gans cross-section not currently supported for KappaExt')
        return

    print(cm.citation)

    if cm.cmtype == 'Graphite':
        qext_pe = scatm.Qext(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='perp'))
        qext_pa = scatm.Qext(a=a, E=E, cm=cmi.CmGraphite(size=cm.size, orient='para'))
        qext = (qext_pa + 2.0*qext_pe) / 3.0
    else:
        qext = scatm.Qext(a=a, E=E, cm=cm)

    if qval:
        return qext
    else:
        cgeo  = np.pi * np.power(a*c.micron2cm, 2)
        return qext * cgeo

def kappa_sca(E, scatm=sms.RGscat(), cm=cmi.CmDrude(), dist=Powerlaw()):
    """
    Opacity to scattering [g^-1 cm^2] integrated over dust grain size distribution.
        |
        | **INPUTS**
        | E     : scalar or np.array : keV
        |
        | **KEYWORDS**
        | scatm : scatmodel object : scatmodels.RGscat (default)
        | cm    : complex index of refraction object : distlib.composition.cmindex.CmDrude (default)
        | dist  : distlib sizedist object : distlib.Powerlaw (default)
    """
    print(cm.citation)
    MD    = 1.0
    ndens = dist.ndens(MD)

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
        kappa = ndens * qsca * cgeo / MD
    else:
        kappa = np.array([])
        for j in range(np.size(E)):
            kappa = np.append(kappa,
                              c.intz(dist.a, ndens * qsca[j,:] * cgeo) / MD)

    return kappa


def kappa_ext(E, scatm=sms.RGscat(), cm=cmi.CmDrude(), dist=Powerlaw()):
    """
    Opacity to EXTINCTION [g^-1 cm^2] integrated over dust grain size distribution
        |
        | **INPUTS**
        | E     : scalar or np.array : keV
        |
        | **KEYWORDS**
        | scatm : scatmodel object : scatmodels.RGscat (default)
        | cm    : complex index of refraction object : distlib.composition.cmindex.CmDrude (default)
        | dist  : distlib sizedist object : distlib.Powerlaw (default)
    """
    # Check if using the RGscat model, which does not do absorption
    if scatm.stype == 'RGscat':
        print 'Rayleigh-Gans cross-section not currently supported for KappaExt'
        kappa = None
        return kappa

    print(cm.citation)
    MD    = 1.0
    ndens = dist.ndens(MD)

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
        kappa = ndens * qext * cgeo / MD
    else:
        kappa = np.array([])
        for j in range(np.size(E)):
            kappa = np.append(kappa,
                              c.intz(dist.a, ndens * qext[j,:] * cgeo) / MD)

    return kappa
