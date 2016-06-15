
import numpy as np

from scipy.special import erf
from scipy.special import gammaincc
from scipy.special import gamma
from scipy.special import expi

from .halodict import HaloDict
from .halo import Halo
from . import galhalo as GH
from ..extinction import sigma_scat as ss

from .. import distlib

__all__ = ['screen_eq','uniform_eq','set_htype']

#--------------------------------------------
# http://www.johndcook.com/gamma_python.html
#
# See ~/Academic/notebooks/test_functions.ipynb for testing
# and
# http://mathworld.wolfram.com/IncompleteGammaFunction.html for general info


def gammainc_fun(a, z):
    """
    Gamma Incomplete function, support for calculating halos analytically
    """
    if np.any(z < 0):
        print('ERROR: z must be >= 0')
        return
    if a == 0:
        return -expi(-z)
    elif a < 0:
        return (gammainc_fun(a+1,z) - np.power(z,a) * np.exp(-z)) / a
    else:
        return gammaincc(a,z) * gamma(a)

def set_htype(halo, xg=None):
    """
    Sets galactic ISM htype values for Halo object
    halo : halo.Halo object
    xg   : float [0-1] : Position of screen where 0 = point source, 1 = observer
        - if None, htype set to 'Uniform'
        - otherwise, hytype set to 'Screen'
    """
    if halo.htype is not None:
        print('WARNING: Halo already has an htype. Overwriting now')
    if xg is None:
        halo.htype = GH.GalHalo(ismtype='Uniform')
    else:
        halo.htype = GH.GalHalo(xg=xg, ismtype='Screen')

    halo.taux  = ss.kappa_sca(E=halo.energy, scatm=halo.gpop.scatm, dist=halo.gpop.sizedist) * halo.gpop.md
    return

#--------------------------------------------
# Screen case ISM

def G_p(halo):
    """
    Returns integral_a0^a1 a^(4-p) da
    """
    a0 = halo.gpop.a[0]
    a1 = halo.gpop.a[-1]
    p  = halo.gpop.sizedist.p
    if p == 5:
        return np.log(a1/a0)
    else:
        return 1.0/(5.0-p) * (np.power(a1,5.0-p) - np.power(a0,5.0-p))

def G_s(halo):
    """
    Function used for evaluating halo from power law distribution of grain sizes (Screen case)
    """
    a0 = halo.gpop.a[0]
    a1 = halo.gpop.a[-1]
    p  = halo.gpop.sizedist.p

    if type(halo) == Halo:
        energy, alpha = halo.energy, halo.alpha
    if type(halo) == HaloDict:
        energy, alpha = halo.superE, halo.superA

    charsig0 = 1.04 * 60.0 / energy
    pfrac    = (7.0-p)/2.0
    const    = alpha**2/(2.0*charsig0**2*halo.htype.xg**2)
    gamma1   = gammainc_fun(pfrac, const * a1**2)
    gamma0   = gammainc_fun(pfrac, const * a0**2)
    return -0.5 * np.power(const, -pfrac) * (gamma1 - gamma0)

def screen_eq(halo, xg=0.5, verbose=False):
    """
    Analytic function for a screen of dust particles from parameters set in halo (taux, a0, a1, p, xg)
        |
        | **MODIFIES**
        | halo.intensity : np.array [arcsec^-2] : I_h/F_a
        | halo.htype
        |
        | **INPUTS**
        | halo : halos.Halo object
        | xg   : float (0.5) : parameterized location of dust screen
        |     [unitless, 0 = location of observer, 1 = location of source]
        | verbose : boolean (False) : prints some information
    """
    test = halo.gpop.sizedist
    if not isinstance(test, distlib.Powerlaw) or isinstance(test, distlib.Grain):
        print('ERROR!! Can only run on power law distribution of grain sizes')
        return
    else:
        set_htype(halo, xg=xg)
        hfrac, energy, alpha = np.array([]), np.array([]), np.array([])

        if type(halo) == Halo:
            if verbose: print('Using a Halo object')
            hfrac = halo.taux
            energy, alpha = halo.energy, halo.alpha

        if type(halo) == HaloDict:
            if verbose: print('Using a halo dictionary')
            NE, NA = len(halo.energy), len(halo.alpha)
            hfrac  = np.tile(halo.taux.reshape(NE,1), NA)  # NE x NA
            energy, alpha = halo.superE, halo.superA  # NE x NA

        if np.size(halo.gpop.a) == 1:
            if verbose: print 'Using a dust grain'
            charsig = 1.04 * 60. / halo.dist.a  / energy  # arcsec
            gterm  = np.exp(-alpha**2 / (2 * charsig**2 * xg**2))
            result = hfrac * gterm / (xg**2 * 2.0*np.pi*charsig**2)
        else:
            if verbose: print 'Using a distribution of grain sizes'
            charsig0 = 1.04 * 60.0 / energy
            const = hfrac / (2.0*np.pi*charsig0**2)
            result = const / xg**2 * G_s(halo) / G_p(halo)

        halo.intensity = result
    return

#--------------------------------------------
# Uniform case ISM

def G_u(halo):
    """
    Function used for evaluating halo from power law distribution of grain sizes
    (Uniform case)
    """
    a0 = halo.gpop.a[0]
    a1 = halo.gpop.a[-1]
    p  = halo.gpop.sizedist.p

    if type(halo) == Halo:
        energy, alpha = halo.energy, halo.alpha
    if type(halo) == HaloDict:
        energy, alpha = halo.superE, halo.superA

    power = 6.0 - p
    pfrac = (7.0-p) / 2.0
    charsig = 1.04 * 60.0 / energy
    const   = alpha / charsig / np.sqrt(2.0)

    A1 = np.power(a1,power) * (1 - erf(const*a1))
    A0 = np.power(a0,power) * (1 - erf(const*a0))
    B1 = np.power(const,-power) * gammainc_fun(pfrac, const**2 * a1**2) / np.sqrt(np.pi)
    B0 = np.power(const,-power) * gammainc_fun(pfrac, const**2 * a0**2) / np.sqrt(np.pi)
    return ((A1-B1) - (A0-B0)) / power

def uniform_eq(halo, verbose=False):
    """
    Analytic function for a uniform distribution of dust particles from parameters set in halo (taux, a0, a1, p)
        |
        | **MODIFIES**
        | halo.intensity : np.array [arcsec^-2] : I_h/F_a
        | halo.htype
        |
        | **INPUTS**
        | halo : halos.Halo object
        | verbose : boolean (False) : boolean (False) : prints some information
    """
    test = halo.gpop.sizedist
    if not isinstance(test, distlib.Powerlaw) or isinstance(test, distlib.Grain):
        print('ERROR!! Can only run on power law distribution of grain sizes')
        return
    else:
        set_htype(halo, xg=None)
        if type(halo) == Halo:
            if verbose: print('Using a Halo object')
            hfrac = halo.taux
            energy, alpha = halo.energy, halo.alpha

        if type(halo) == HaloDict:
            if verbose: print('Using a halo dictionary')
            NE, NA = len(halo.energy), len(halo.alpha)
            hfrac = np.tile(halo.taux.reshape(NE,1), NA)  # NE x NA
            energy, alpha = halo.superE, halo.superA  # NE x NA

        if np.size(halo.gpop.a) == 1:
            if verbose: print('Using a dust grain')
            charsig = 1.04 * 60. / halo.dist.a  / energy  # arcsec
            eterm  = 1 - erf(alpha / charsig / np.sqrt(2.))
            result = hfrac * eterm * np.sqrt(np.pi/2.0) / (2.0*np.pi*charsig*alpha)
        else:
            if verbose: print('Using a distribution of grain sizes')
            charsig = 1.04 * 60.0 / energy
            const = hfrac / (alpha * charsig * np.sqrt(8.0*np.pi))
            result = const * G_u(halo) / G_p(halo)

        halo.intensity = result
    return
