#! /Library/Frameworks/EPD64.framework/Versions/Current/bin/python

import numpy as np
import matplotlib.pyplot as plt

import galhalo as GH
import halodict as HD
import analytic as AH

from scipy.interpolate import interp1d

import sys
sys.argv

## UPDATED June 11, 2013 : I want to make halo_lib 
## independent of asciidata, radprofile, and aeff

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
ALPHA  = np.power( 10.0, np.arange(-1.0,3.01,0.1) )

AEFF   = HD.aeff( 'zeroth_aeff_cycle7.dat' )

#---------------------------------------------------------------------
## Set up halo dictionary

def get_spec( specfile ):
	"""
	Returns halodict.Spectrum object containing the 
	source flux [photons/cm^2/s] as a function of energy [keV]
	"""
	data = c.read_table( specfile, 2 )
	energy  = np.array( data[0] )
	srcflux = np.array( data[1] )
	return HD.Spectrum( energy, srcflux )

def get_halodict( spectrum, rad=grains, scatm=SCATM, alpha=ALPHA ):
	"""
	Returns halodict.HaloDict object with halodict.Spectrum.bin values as keys
	"""
	return HD.HaloDict( spectrum.bin, rad=rad, scatm=scatm, alpha=alpha )

def analytic_screen( halodict, xg=0.5, NH=NH, d2g=0.009, verbose=False ):
	"""
	Alters halodict with analytic halo model for dust screen
    ----------------------------------------------------
	FUNCTION analytic_screen( halodict, xg=0.5, NH=NH, d2g=0.009, verbose=False )
	RETURNS  : empty 
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    xg       : float [0-1] : position of screen where 0 = point source, 1 = observer
    NH       : float [cm^-2] : Hydrogen column
    d2g      : float : Dust-to-gas mass ratio
    verbose  : boolean : If true, print halo energy at each calculation step
	"""
	print 'Calculating analytic halo model for a dust screen at x =', xg
	for h in halodict:
		if verbose:
			print 'Calculating halo energy :', h.E0, ' keV'
		AH.set_htype( h, xg=xg, NH=NH, d2g=d2g )
		AH.screen_eq( h )
	return
	
def analytic_uniform( halodict, NH=NH, d2g=0.009, verbose=False ):
	"""
	Alters halodict with analytic halo model for uniform dust distribution
    ----------------------------------------------------
	FUNCTION analytic_uniform( halodict, NH=NH, d2g=0.009, verbose=False )
	RETURNS  : empty 
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    NH       : float [cm^-2] : Hydrogen column
    d2g      : float : Dust-to-gas mass ratio
    verbose  : boolean : If true, print halo energy at each calculation step
	"""
	print 'Calculating analytic halo model for uniform ISM'
	for h in halodict:
		if verbose:
			print 'Calculating halo energy :', h.E0, ' keV'
		AH.set_htype( h, NH=NH, d2g=d2g )
		AH.uniform_eq( h )
	return

def screen( halodict, xg=0.5, NH=NH, d2g=0.009, verbose=False ):
	"""
	Performs numerical integration dust screen calculation with each halo in halodict
    ----------------------------------------------------
	FUNCTION screen( halodict, xg=0.5, NH=NH, d2g=0.009, verbose=False )
	RETURNS  : empty 
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    xg       : float [0-1] : position of screen where 0 = point source, 1 = observer
    NH       : float [cm^-2] : Hydrogen column
    d2g      : float : Dust-to-gas mass ratio
    verbose  : boolean : If true, print halo energy at each calculation step
	"""
	print 'Numerically integrating halo model for a dust screen at x =', xg
	for h in halodict:
		if verbose:
			print 'Calculating halo energy :', h.E0, ' keV'
		GH.DiscreteISM( h, xg=xg, NH=NH, d2g=d2g )
	return

def uniform( halodict, NH=NH, d2g=0.009, verbose=False ):
	"""
	Performs numerical integration with uniform dust distribution
		calculation with each halo in halodict
    ----------------------------------------------------
	FUNCTION uniform( halodict, NH=NH, d2g=0.009, verbose=False )
	RETURNS  : empty 
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    NH       : float [cm^-2] : Hydrogen column
    d2g      : float : Dust-to-gas mass ratio
    verbose  : boolean : If true, print halo energy at each calculation step
	"""
	print 'Numerically integrating halo model for uniform ISM'
	for h in halodict:
		if verbose:
			print 'Calculating halo energy:', h.E0, ' keV'
		GH.UniformISM( h, NH=NH, d2g=d2g )
	return

#---------------------------------------------------------------------
## Use the corrected flux [I_app = I_PS * exp(tau)] to find the total
## halo brightness

def totalhalo( halodict, spectrum ):
	"""
	Alters halodict by running halodict.HaloDict.total_halo( corrflux )
    ----------------------------------------------------
	FUNCTION totalhalo( halodict, spectrum )
	RETURNS  : np.array : Corrected flux before scattering (F_a)
		assuming F_PS = F_a exp(-tau)
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    spectrum : halodict.Spectrum object
	"""
	corrflux = spectrum.flux * np.exp( halodict.taux )
	halodict.total_halo( corrflux )
	return corrflux

#---------------------------------------------------------------------
## April 8, 2013 : Write a simple function wrap this all up?

EXPOSURE = 50000.  # default 50ks exposure

def simulate_surbri( halodict, spectrum, aeff, exposure=EXPOSURE ):
	'''
	Take a halo dictionary with a simulated halo, 
	and simulate a surface brightness profile with it.
    ----------------------------------------------------
    FUNCTION simulate_surbri( halodict, spectrum, aeff, exposure=50000.0 )
    RETURNS : scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    ----------------------------------------------------
    halodict : halodict.HaloDict object
    spectrum : halodict.Spectrum object
    aeff     : interp1d object : x = energy [keV], y = effective area [cm^2]
    exposure : float : Exposure time [sec]
	'''
	flux_dict = dict( zip( spectrum.bin, spectrum.flux ) )
	arcsec2pix = 0.5  #arcsec/pix
	result = 0.0

	for ee in halodict.energy:
		corr_counts = flux_dict[ee] * np.exp( halodict[ee].taux ) * aeff(ee) * EXPOSURE
		halo_counts = corr_counts * halodict[ee].intensity * arcsec2pix**2 # pixels^-2
		result = result + halo_counts

	return interp1d( halodict.alpha/arcsec2pix, result )

def simulate_screen( specfile, a0=0.1, a1=None, p=3.5, NH=NH, d2g=0.009, xg=0.5, \
	alpha=ALPHA, aeff=AEFF, exposure=EXPOSURE ):
	'''
	Simulate a surface brightness profile from spectrum file
	for a screen of dust at xg, using 3-5 free parameters
    ----------------------------------------------------
    FUNCTION simulate_screen( specfile, a0=0.1, a1=None, p=3.5, d2g=0.009, xg=0.5, \
    	alpha=ALPHA, aeff=AEFF, exposure=EXPOSURE )
    RETURNS : scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    ----------------------------------------------------
    specfile : string : Name of spectrum file
    a0       : float [um] : Minimum (or single) grain size to use
    a1       : float [um] : Maximum grain size for distribution (if None, single used)
    p        : float : Power law index for grain size distribution
    d2g      : float : Dust-to-gas mass ratio
    xg       : float [0-1] : Position of screen where 0 = point source, 1 = observer
    alpha    : np.array [arcsec] : Angles for halo intensity values
    aeff     : intper1d object : x = energy [keV], y = effective area [cm^2]
    exposure : float [sec] : Observation exposure time
	'''
	source_flux = get_spec( specfile )
	if a1 == None:
		dust_dist = GH.dust.Grain( rad=a0 )
	else:
		dust_dist = GH.dust.Dustdist( p=p, rad=GH.dust.adist(a0,a1) )
	
	halo_dict = get_halodict( source_flux, rad=dust_dist, scatm=SCATM, alpha=alpha )
	analytic_screen( halo_dict, xg=xg, NH=NH, d2g=d2g )
	result = simulate_surbri( halo_dict, source_flux, aeff, exposure=exposure )

	return result

def simulate_uniform( specfile, a0=0.1, a1=None, p=None, NH=NH, d2g=0.009, \
	alpha=ALPHA, aeff=AEFF, exposure=EXPOSURE ):
	'''
	Simulate a surface brightness profile from spectrum file
	for a uniform distribution of dust, using 2-4 free parameters
    ----------------------------------------------------
    FUNCTION simulate_screen( specfile, a0=0.1, a1=None, p=3.5, d2g=0.009, xg=0.5, \
    	alpha=ALPHA, aeff=AEFF, exposure=EXPOSURE )
    RETURNS : scipy.interpolate.interp1d object : x = pixels, y = counts/pix^2
    ----------------------------------------------------
    specfile : string : Name of spectrum file
    a0       : float [um] : Minimum (or single) grain size to use
    a1       : float [um] : Maximum grain size for distribution (if None, single used)
    p        : float : Power law index for grain size distribution
    d2g      : float : Dust-to-gas mass ratio
    alpha    : np.array [arcsec] : Angles for halo intensity values
    aeff     : intper1d object : x = energy [keV], y = effective area [cm^2]
    exposure : float [sec] : Observation exposure time
	'''
	source_flux = get_spec( specfile )
	if a1 == None:
		dust_dist = GH.dust.Grain( rad=a0 )
	else:
		dust_dist = GH.dust.Dustdist( p=p, rad=GH.dust.adist(a0,a1) )
	
	halo_dict = get_halodict( source_flux, rad=dust_dist, scatm=SCATM, alpha=alpha )
	analytic_uniform( halo_dict, NH=NH, d2g=d2g )
	result = simulate_surbri( halo_dict, source_flux, aeff, exposure=exposure )
	
	return result

