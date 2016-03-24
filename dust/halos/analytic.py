
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.integrate import trapz

## REQUIRES Halo objects to work
from .halodict import HaloDict
from .halo import Halo

from . import galhalo as GH
from .. import distlib

## UPDATED June 11, 2013 : Make the halos calculation faster for halodicts?

## UPDATED May 27, 2013  : Rewrote gammainc_fun function to be more robust
## UPDATED April 4, 2013 : To treat halos in same way as DiscreteISM
## 							and UniformISM functions in galhalo.py
## CREATED April 3, 2013

#--------------------------------------------
# http://www.johndcook.com/gamma_python.html
#
# See ~/Academic/notebooks/test_functions.ipynb for testing
# and
# http://mathworld.wolfram.com/IncompleteGammaFunction.html for general info

from scipy.special import gammaincc
from scipy.special import gamma
from scipy.special import expi

def gammainc_fun( a, z ):
    if np.any(z < 0):
        print('ERROR: z must be >= 0')
        return
    if a == 0:
        return -expi(-z)
    elif a < 0:
        return ( gammainc_fun(a+1,z) - np.power(z,a) * np.exp(-z) ) / a
    else:
        return gammaincc(a,z) * gamma(a)

def set_htype( halo, plaw=distlib.Powerlaw(), xg=None, NH=1.0e20, d2g=0.009 ):
    '''
    Sets galactic ISM htype values for Halo object
    --------------------------------------------------------------
    FUNCTION set_htype( halo, xg=None, NH=1.0e20, d2g=0.009 )
    RETURN : empty
    --------------------------------------------------------------
    halo : halo.Halo object
    xg  : float [0-1] : Position of screen where 0 = point source, 1 = observer
        - if None, htype set to 'Uniform'
        - otherwise, hytype set to 'Screen'
    NH  : float [cm^-2] : Hydrogen column density
    d2g : float : Dust-to-gas mass ratio
    '''
    if halo.htype != None:
        print('WARNING: Halo already has an htype. Overwriting now')
    if xg == None:
        halo.htype = GH.GalHalo( NH=NH, d2g=d2g, ismtype='Uniform' )
    else:
        halo.htype = GH.GalHalo( xg=xg, NH=NH, d2g=d2g, ismtype='Screen' )
    md         = NH * GH.c.m_p * d2g
    halo.plaw  = plaw
    dspec      = distlib.DustSpectrum()
    dspec.calc_from_dist(halo.plaw, md=md)
    halo.dist  = dspec
    halo.taux  = GH.ss.KappaScat(E=halo.energy, scatm=halo.scatm, dist=halo.dist ).kappa * halo.dist.md
    return

#--------------------------------------------
## Screen case ISM

def G_p( halo ):
    '''
    Returns integral_a0^a1 a^(4-p) da
    '''
    a0 = halo.dist.a[0]
    a1 = halo.dist.a[-1]
    p  = halo.plaw.p
    if p == 5:
        return np.log( a1/a0 )
    else:
        return 1.0/(5.0-p) * ( np.power(a1,5.0-p) - np.power(a0,5.0-p) )

def G_s( halo ):
    '''
    Function used for evaluating halo from power law distribution of grain sizes (Screen case)
    '''
    a0 = halo.dist.a[0]
    a1 = halo.dist.a[-1]
    p  = halo.plaw.p

    if type(halo) == Halo:
        energy, alpha = halo.energy, halo.alpha
    if type(halo) == HaloDict:
        energy, alpha = halo.superE, halo.superA

    charsig0 = 1.04 * 60.0 / energy
    pfrac    = (7.0-p)/2.0
    const    = alpha**2/(2.0*charsig0**2*halo.htype.xg**2)
    gamma1   = gammainc_fun( pfrac, const * a1**2 )
    gamma0   = gammainc_fun( pfrac, const * a0**2 )
    return -0.5 * np.power( const, -pfrac ) * ( gamma1 - gamma0 )

def screen_eq( halo, xg=0.5, verbose=False, **kwargs ):
    '''
    Analytic function for a screen of dust particles
        from parameters set in halo (taux, a0, a1, p, xg)

    | **MODIFIES**
    | halo.htype, halo.intensity

    | **INPUTS**
    | halo : halo.Halo object
    | xg   : float (0.5) : parameterized location of dust screen
    |     [unitless, 0 = location of observer, 1 = location of source]
    | verbose : boolean (False) : prints some information

    | **RETURNS**
    | np.array [arcsec^-2] : I_h/F_a
    '''

    set_htype( halo, xg=xg, **kwargs )
    hfrac, energy, alpha = np.array([]), np.array([]), np.array([])

    if type(halo) == Halo:
        if verbose: print 'Using a Halo object'
        hfrac = halo.taux
        energy, alpha = halo.energy, halo.alpha

    if type(halo) == HaloDict:
        if verbose: print 'Using a halo dictionary'
        NE, NA = len(halo.energy), len(halo.alpha)
        hfrac  = np.tile( halo.taux.reshape(NE,1), NA ) # NE x NA
        energy, alpha = halo.superE, halo.superA # NE x NA

    if np.size(halo.dist.a) == 1:
        if verbose: print 'Using a dust grain'
        charsig = 1.04 * 60. / halo.dist.a  / energy  #arcsec
        gterm  = np.exp( -alpha**2 / (2 * charsig**2 * xg**2) )
        result = hfrac * gterm / ( xg**2 * 2.0*np.pi*charsig**2 )
    else:
        if verbose: print 'Using a distribution of grain sizes'
        charsig0 = 1.04 * 60.0 / energy
        const = hfrac / ( 2.0*np.pi*charsig0**2 )
        result = const / xg**2 * G_s(halo) / G_p(halo)

    halo.intensity = result
    return

#--------------------------------------------
## Uniform case ISM

def G_u( halo ):
    '''
    Function used for evaluating halo from power law distribution of grain sizes
    (Uniform case)
    '''
    a0 = halo.dist.a[0]
    a1 = halo.dist.a[-1]
    p  = halo.plaw.p

    if type(halo) == Halo:
        energy, alpha = halo.energy, halo.alpha
    if type(halo) == HaloDict:
        energy, alpha = halo.superE, halo.superA

    power = 6.0 - p
    pfrac = (7.0-p) / 2.0
    charsig = 1.04 * 60.0 / energy
    const   = alpha / charsig / np.sqrt(2.0)

    A1 = np.power(a1,power) * ( 1 - erf(const*a1) )
    A0 = np.power(a0,power) * ( 1 - erf(const*a0) )
    B1 = np.power(const,-power) * gammainc_fun( pfrac, const**2 * a1**2 ) / np.sqrt(np.pi)
    B0 = np.power(const,-power) * gammainc_fun( pfrac, const**2 * a0**2 ) / np.sqrt(np.pi)
    return ( (A1-B1) - (A0-B0) ) / power

def uniform_eq( halo, verbose=False, **kwargs ):
    '''
    Analytic function for a uniform distribution of dust particles
        from parameters set in halo (taux, a0, a1, p)

    | **MODIFIES**
    | halo.htype, halo.intensity

    | **INPUTS**
    | halo : halo.Halo object
    | verbose : boolean (False) : boolean (False) : prints some information

    | **RETURNS**
    | np.array [arcsec^-2] : I_h/F_a
    '''

    set_htype( halo, xg=None, **kwargs )

    if type(halo) == Halo:
        if verbose: print 'Using a Halo object'
        hfrac = halo.taux
        energy, alpha = halo.energy, halo.alpha

    if type(halo) == HaloDict:
        if verbose: print 'Using a halo dictionary'
        NE, NA = len(halo.energy), len(halo.alpha)
        hfrac = np.tile( halo.taux.reshape(NE,1), NA ) # NE x NA
        energy, alpha = halo.superE, halo.superA # NE x NA

    if np.size(halo.dist.a) == 1:
        if verbose: print 'Using a dust grain'
        charsig = 1.04 * 60. / halo.dist.a  / energy  #arcsec
        eterm  = 1 - erf( alpha / charsig / np.sqrt(2.) )
        result = hfrac * eterm * np.sqrt(np.pi/2.0) / (2.0*np.pi*charsig*alpha)
    else:
        if verbose: print 'Using a distribution of grain sizes'
        charsig = 1.04 * 60.0 / energy
        const = hfrac / ( alpha * charsig * np.sqrt(8.0*np.pi) )
        result = const * G_u(halo) / G_p(halo)

    halo.intensity = result
    return
