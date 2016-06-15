#! /Library/Frameworks/EPD64.framework/Versions/Current/bin/python

import numpy as np
from scipy.interpolate import interp1d

from .. import distlib
from . import galhalo as GH
from . import halodict as HD
from . import analytic as AH

from astropy.io import ascii

#---------------------------------------------------------------------
# Set up the model
ALPHA  = np.logspace(0.0, 3.0, 30)
GPOP   = distlib.make_MRN_grainpop()

#---------------------------------------------------------------------

def screen(halodict, xg=0.5, verbose=False):
    """
    | Performs numerical integration dust screen calculation with each halo in halodict
    |
    | **MODIFIES**
    | halodict.intensity, halodict.htype
    |
    | **INPUTS**
    | halodict : halodict.HaloDict object
    | xg       : float [0-1] : position of screen where 0 = point source, 1 = observer
    | verbose  : boolean : If true, print halo energy at each calculation step
    """
    print('Numerically integrating halo model for a dust screen at x =', xg)
    AH.set_htype(halodict, xg=xg)
    for i in range(halodict.len):
        if verbose: print('Calculating halo energy :', halodict.energy[i], 'keV')
        halo_temp = GH.Halo(halodict.energy[i], alpha=halodict.alpha, gpop=halodict.gpop)
        GH.screenISM(halo_temp, xg=xg)
        halodict.intensity[i,:] = halo_temp.intensity
    return

def uniform(halodict, verbose=False):
    """
    | Performs numerical integration with uniform dust distribution calculation with each halo in halodict
    |
    | **MODIFIES**
    | halodict.intensity, halodict.htype
    |
    | **INPUTS**
    | halodict : halodict.HaloDict object
    | verbose  : boolean : If true, print halo energy at each calculation step
    """
    print('Numerically integrating halo model for uniform ISM')
    AH.set_htype(halodict)
    for i in range(halodict.len):
        if verbose: print('Calculating halo energy:', halodict.energy[i], 'keV')
        halo_temp = GH.Halo(halodict.energy[i], alpha=halodict.alpha, gpop=halodict.gpop)
        GH.uniformISM(halo_temp)
        halodict.intensity[i,:] = halo_temp.intensity
    return

'''
# 2016.06.10 - deprecated
#---------------------------------------------------------------------
# Use the corrected flux [I_app = I_PS * exp(tau)] to find the total
# halo brightness

def totalhalo(halodict, spectrum):
    """
    | Alters halodict by running halodict.HaloDict.total_halo( corrflux )
    |
    | **MODIFIES**
    | halodict.total
    |
    | **INPUTS**
    | halodict : halodict.HaloDict object
    | spectrum : flux for the energy values associated with halodict
    |
    | **RETURNS**
    | np.array : Corrected flux before scattering (F_a)
    |     assuming F_PS = F_a exp(-tau)
    """
    corrflux = spectrum * np.exp(halodict.taux)
    halodict.total_halo(corrflux)
    return corrflux
'''
#---------------------------------------------------------------------

def simulate_intensity(halodict, spectrum):
    '''
    Take a halo dictionary with an evaluated profile, and simulate a surface brightness profile with it.
        |
        | **INPUTS**
        | halodict : halodict.HaloDict object
        | spectrum : corrected flux for each energy value in halodict
        |
        | **RETURNS**
        | scipy.interpolate.interp1d object : x = arcsec, y = flux/arcsec^2
    '''
    result = 0.0

    corr_flux = spectrum * np.exp(halodict.taux)

    NE, NA = halodict.len, halodict.hsize
    halo_flux = np.tile(corr_flux.reshape(NE,1), NA) * halodict.intensity
    # flux/arcsec^2

    result = np.sum(halo_flux, 0)
    return interp1d(halodict.alpha, result)  # arcsec, flux/arcsec^2

def simulate_screen(specfile, xg=0.5, alpha=ALPHA, gpop=GPOP,
                    elim=None, return_dict=False, verbose=False):
    '''
    | Simulate a surface brightness profile from spectrum file for a screen of dust at xg, using 3-5 free parameters
    |
    | **INPUTS**
    | specfile : string : Name of spectrum file
    | xg       : float [0-1] : Position of screen where 0 = point source, 1 = observer
    | alpha    : np.array [arcsec] : Angles for halo intensity values
    | gpop     : distlib.GrainPop object
    | elim     : tuple containing energy limits [keV]
    | return_dict : boolean (False) : if True, returns halodict instead of interp object
    | verbose  : boolean (False) : prints extra information if True
    |
    | **RETURNS**
    | if return_dict == False :
    |     scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    | if return_dict == True :
    |     HaloDict object with full benefits of information
    '''
    data = ascii.read(specfile)
    energy, flux = data['col1'], data['col2']

    ii = range(len(energy))
    if elim is not None:
        if verbose: print('Limiting energy to values between %f and %f keV' % elim)
        ii = (energy >= elim[0]) & (energy <= elim[1])

    halo_dict = HD.HaloDict(energy[ii], alpha=alpha, gpop=gpop)
    AH.screen_eq(halo_dict, xg=xg)
    result = simulate_intensity(halo_dict, flux[ii])

    if return_dict: return halo_dict
    else: return result

def simulate_uniform(specfile, alpha=ALPHA, gpop=GPOP,
                     elim=None, return_dict=False, verbose=False):
    '''
    | Simulate a surface brightness profile from spectrum file for a uniform distribution of dust, using 2-4 free parameters
    |
    | **INPUTS**
    | specfile : string : Name of spectrum file
    | alpha    : np.array [arcsec] : Angles for halo intensity values
    | gpop     : distlib.GrainPop object
    | elim     : tuple containing energy limits [keV]
    | return_dict : boolean (False) : if True, returns halodict instead of interp object
    | verbose  : boolean (False) : prints extra information if True
    |
    | **RETURNS**
    | if return_dict == False :
    |     scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    | if return_dict == True :
    |     HaloDict object with full benefits of information
    '''
    data = ascii.read(specfile)
    energy, flux = data['col1'], data['col2']

    ii = range(len(energy))
    if elim is not None:
        if verbose: print('Limiting energy to values between %f and %f keV' % elim)
        ii = (energy >= elim[0]) & (energy <= elim[1])

    halo_dict = HD.HaloDict(energy[ii], alpha=alpha, gpop=gpop)
    AH.uniform_eq(halo_dict)
    result = simulate_intensity(halo_dict, flux[ii])

    if return_dict: return halo_dict
    else: return result
