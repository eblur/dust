
import numpy as np

def prop_add( xerr=0.0, yerr=0.0 ):
	return np.sqrt( xerr**2 + yerr**2 )

def prop_div( x, y, xerr=0.0, yerr=0.0 ):
	F = x / y
	return np.sqrt( xerr**2 + F**2 * yerr**2 ) / y

def prop_mult( x, y, xerr=0.0, yerr=0.0 ):
	F = x * y
	return np.sqrt( (xerr/x)**2 + (yerr/y)**2 ) * F
