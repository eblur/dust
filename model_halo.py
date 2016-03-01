#! /Library/Frameworks/EPD64.framework/Versions/Current/bin/python

import numpy as np

import galhalo as GH
import halodict as HD
import analytic as AH
import constants as c

from scipy.interpolate import interp1d
from scipy import logical_and
## Bizzare -- I had to import logical_and for some reason

## December 11, 2014 : Remove dependence on AEFF, which came from PIMMS
##    Instead of simulating counts/pix^2, using raw flux/arcsec^2

## UPDATED June 11, 2013 : I want to make halo_lib 
## independent of asciidata, radprofile, and aeff
## CAVEAT : Path to effective area data file needs to be specified

## UPDATED April 4, 2013 : To include analytic solutions for galactic ISM
## (RG Gans + Drude solution only)

## CREATED August 29, 2012 : Read spectrum from a text file with same
## format as that required for ChaRT simulation.  (Columns are energy
## [keV] and flux [photons/s/cm**2], tab separated.)  Then use this to
## simulate a halo.  I will set the parameters here, but in the future
## it would be nice to read them from a parameter file or using some
## other flexible input.

#---------------------------------------------------------------------
## Model parameters : As of Aug 29, these come from Predehl & Schmitt 1995
## April 4, 2013 : Updated from more recent data
P    = 3.5
AMIN = 0.05
AMAX = 0.3
NH   = 3.6e22

## Set up the model
## MAGIC NUMBERS are in ALLCAPS
NA     = 50
da     = ( np.log10(AMAX) - np.log10(AMIN) ) / NA
avals  = np.power( 10.0, np.arange( np.log10(AMIN), np.log10(AMAX)+da, da ) )
grains = GH.dust.Dustdist( p=P, rad=avals )

SCATM  = GH.ss.Scatmodel()
ALPHA  = np.power( 10.0, np.arange(0.0,3.01,0.1) )

#---------------------------------------------------------------------

def screen( halodict, xg=0.5, NH=NH, d2g=0.009, verbose=False ):
    """
    Performs numerical integration dust screen calculation with each halo in halodict

    | **MODIFIES**
    | halodict.intensity, halodict.htype
    
    | **INPUTS**
    | halodict : halodict.HaloDict object
    | xg       : float [0-1] : position of screen where 0 = point source, 1 = observer
    | NH       : float [cm^-2] : Hydrogen column
    | d2g      : float : Dust-to-gas mass ratio
    | verbose  : boolean : If true, print halo energy at each calculation step
    """
    print 'Numerically integrating halo model for a dust screen at x =', xg
    AH.set_htype( halodict, xg=xg, NH=NH, d2g=d2g )
    for i in range( halodict.len ):
        if verbose: print 'Calculating halo energy :', halodict.energy[i], ' keV'
        halo_temp = GH.Halo( halodict.energy[i], alpha=halodict.alpha, \
            scatm=halodict.scatm, rad=halodict.rad )
        GH.DiscreteISM( halo_temp, xg=xg, NH=NH, d2g=d2g )
        halodict.intensity[i,:] = halo_temp.intensity
    return

def uniform( halodict, NH=NH, d2g=0.009, verbose=False ):
    """
    Performs numerical integration with uniform dust distribution
        calculation with each halo in halodict
    
    | **MODIFIES**
    | halodict.intensity, halodict.htype
    
    | **INPUTS**
    | halodict : halodict.HaloDict object
    | NH       : float [cm^-2] : Hydrogen column
    | d2g      : float : Dust-to-gas mass ratio
    | verbose  : boolean : If true, print halo energy at each calculation step
    """
    print 'Numerically integrating halo model for uniform ISM'
    AH.set_htype( halodict, NH=NH, d2g=d2g )
    for i in range( halodict.len ):
        if verbose: print 'Calculating halo energy:', halodict.energy[i], ' keV'
        halo_temp = GH.Halo( halodict.energy[i], alpha=halodict.alpha, \
            scatm=halodict.scatm, rad=halodict.rad )
        GH.UniformISM( halo_temp, NH=NH, d2g=d2g )
        halodict.intensity[i,:] = halo_temp.intensity
    return

#---------------------------------------------------------------------
## Use the corrected flux [I_app = I_PS * exp(tau)] to find the total
## halo brightness

def totalhalo( halodict, spectrum ):
    """
    Alters halodict by running halodict.HaloDict.total_halo( corrflux )
    
    | **MODIFIES**
    | halodict.total

    | **INPUTS**
    | halodict : halodict.HaloDict object
    | spectrum : flux for the energy values associated with halodict

    | **RETURNS**
    | np.array : Corrected flux before scattering (F_a)
    |     assuming F_PS = F_a exp(-tau)
    """
    corrflux = spectrum * np.exp( halodict.taux )
    halodict.total_halo( corrflux )
    return corrflux

#---------------------------------------------------------------------

def simulate_intensity( halodict, spectrum ):
    '''
    Take a halo dictionary with an evaluated profile, 
    and simulate a surface brightness profile with it. 
    The arcsec to pixel conversion is based on Chandra.

    | **INPUTS**
    | halodict : halodict.HaloDict object
    | spectrum : corrected flux for each energy value in halodict

    | **RETURNS**
    | scipy.interpolate.interp1d object : x = arcsec, y = flux/arcsec^2
    '''
    arcsec2pix = 0.5  #arcsec/pix (Chandra)
    result = 0.0
    
    corr_flux = spectrum * np.exp( halodict.taux )
    
    NE, NA = halodict.len, halodict.hsize
    halo_flux = np.tile( corr_flux.reshape(NE,1), NA ) * halodict.intensity
    # flux/arcsec^2
    
    result = np.sum( halo_flux, 0 )
    return interp1d( halodict.alpha, result ) # arcsec, flux/arcsec^2

def simulate_screen( specfile, a0=0.05, a1=None, p=3.5, \
    NH=1.0e22, d2g=0.009, xg=0.5, rho=3.0, return_dict=False, \
    alpha=ALPHA, scatm=SCATM, elim=None, na=50, v=False ):
    '''
    Simulate a surface brightness profile from spectrum file
    for a screen of dust at xg, using 3-5 free parameters

    | **INPUTS**
    | specfile : string : Name of spectrum file
    | a0       : float [um] : Minimum (or single) grain size to use (0.05)
    | a1       : float [um] : Maximum grain size for distribution (if None, single used)
    | p        : float : Power law index for grain size distribution
    | NH       : float [cm^-2] : Hyrdogen column (1.0e22)
    | d2g      : float : Dust-to-gas mass ratio (0.009)
    | rho      : float [g cm^-3] : mass density of a dust grain (3)
    | dict     : boolean (False) : if True, returns halodict instead of interp object
    | xg       : float [0-1] : Position of screen where 0 = point source, 1 = observer
    | alpha    : np.array [arcsec] : Angles for halo intensity values
    | scatm    : ss.Scatmodel()
    | elim     : tuple containing energy limits [keV]
    | na       : number of bins to use for grain size distribution

    | **RETURNS**
    | if dict == False : 
    |     scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    | if dict == True :
    |     HaloDict object with full benefits of information
    '''
    energy, flux = HD.get_spectrum( specfile )
    if a1 == None:
        dust_dist = GH.dust.Grain( rad=a0, rho=rho )
    else:
        dth = (a1-a0)/na
        dust_dist = GH.dust.Dustdist( p=p, rad=np.arange(a0,a1+dth,dth), rho=rho )
    
    ii = range( len(energy) )
    if elim != None:
        if v: print 'Limiting energy to values between', elim[0], 'and', elim[1], 'keV'
        ii = np.where( logical_and( energy>=elim[0], energy<=elim[1] ) )[0]
    
    halo_dict = HD.HaloDict( energy[ii], rad=dust_dist, scatm=scatm, alpha=alpha )
    AH.screen_eq( halo_dict, xg=xg, NH=NH, d2g=d2g )
    result = simulate_intensity( halo_dict, flux[ii] )
    
    if return_dict : return halo_dict
    else : return result

def simulate_uniform( specfile, a0=0.1, a1=None, p=3.5, \
    NH=1.0e22, d2g=0.009, rho=3.0, return_dict=False, \
    alpha=ALPHA, scatm=SCATM, elim=None, na=50, v=False ):
    '''
    Simulate a surface brightness profile from spectrum file
    for a uniform distribution of dust, using 2-4 free parameters
    
    | **INPUTS**
    | specfile : string : Name of spectrum file
    | a0       : float [um] : Minimum (or single) grain size to use
    | a1       : float [um] : Maximum grain size for distribution (if None, single used)
    | p        : float : Power law index for grain size distribution
    | NH       : float [cm^-2] : Hyrdogen column (1.0e22)
    | d2g      : float : Dust-to-gas mass ratio (0.009)
    | rho      : float [g cm^-3] : mass density of a dust grain (3)
    | dict     : boolean (False) : if True, returns halodict instead of interp object
    | alpha    : np.array [arcsec] : Angles for halo intensity values
    | aeff     : intper1d object : x = energy [keV], y = effective area [cm^2]
    | exposure : float [sec] : Observation exposure time
    | scatm    : ss.Scatmodel()
    | elim     : tuple containing energy limits [keV]
    | na       : number of bins to use for grain size distribution
    
    | **RETURNS**
    | if dict == False : 
    |     scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    | if dict == True :
    |     HaloDict object with full benefits of information
    '''
    energy, flux = HD.get_spectrum( specfile )
    if a1 == None:
        dust_dist = GH.dust.Grain( rad=a0, rho=rho )
    else:
        dth = (a1-a0)/na
        dust_dist = GH.dust.Dustdist( p=p, rad=np.arange(a0,a1+dth,dth), rho=rho )
    
    ii = range( len(energy) )
    if elim != None:
        if v: print 'Limiting energy to values between', elim[0], 'and', elim[1], 'keV'
        ii = np.where( logical_and( energy>=elim[0], energy<=elim[1] ) )[0]
    
    halo_dict = HD.HaloDict( energy[ii], rad=dust_dist, scatm=scatm, alpha=alpha )
    AH.uniform_eq( halo_dict, NH=NH, d2g=d2g )
    result = simulate_intensity( halo_dict, flux[ii] )
    
    if return_dict : return halo_dict
    else : return result

