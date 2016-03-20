
import math
import numpy as np
import scipy as sp

##----------------------------------------------------------
## Generic constants

# Speed of light
clight = 3.e10 # cm/s

# Planck's h constant
hplanck = np.float64( 4.136e-18 )  # keV s

# Electron radius
r_e = 2.83e-13 # cm

# Mass of proton
m_p = np.float64( 1.673e-24 ) # g

##----------------------------------------------------------
## Constants for converting things

micron2cm = 1.e-6 * 100.0 # cm/micron
pc2cm     = 3.09e18       # cm/pc

arcs2rad  = (2.0*np.pi) / (360.*60.*60.)  # rad/arcsec
arcm2rad  = (2.0*np.pi) / (360.*60.)      # rad/arcmin

const_hc  = (clight * hplanck) # keV cm

##----------------------------------------------------------
## Cosmology related constants

# Hubble's constant
h0 = 75. #km/s/Mpc

# Critical density for Universe
rho_crit = np.float64(1.1e-29)

# Density in units of rho_crit
omega_d = 1.e-5 # dust
omega_m = 0.3   # matter
omega_l = 0.7   # dark energy

# c/H term in distance integral (a distance)
# c/H = Mpc, then convert to cm
cperh0  = (clight * 1.e-5 / h0) * (1.e6 * pc2cm)



def h0():
    return 75.  #km/s/Mpc

def rho_crit():
    return np.float64(1.1e-29)

def omega_d():
    return 1e-5

def omega_m():
    return 0.3

def omega_l():
    return 0.7

def intz(x, y):
    from scipy import integrate
    return sp.integrate.trapz(y,x)
# Note that scipy calls integration in reverse order as I do

def int(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.sum( y[:-1]*dx + 0.5*dx*dy )

def c():
    return 3.e10 # cm/s

def h():
    return np.float64( 4.136e-18 )  # keV s

def re():
    return 2.83e-13  # cm (electron radius)

def mp():
    return np.float64( 1.673e-24 ) # g (mass of proton)

def micron2cm():
    return 1e-6 * 100

def pc2cm():
    return 3.09e18

def cperh0():
    return c()*1.e-5 / h0() * (1.e6 * pc2cm() )  #cm

def kev2lam():
    return 1.240e-7   # cm keV

def arcs2rad():
    return 2.0*np.pi / 360. / 60. / 60.  # rad/arcsec

def arcm2rad():
    return 2.0*np.pi / 360. / 60.        # rad/arcmin


#------- Save and restore functions, similar to IDL -------#
# http://idl2python.blogspot.com/2010/10/save-and-restore-2.html
# Updated April 20, 2012 to store objects
# http://wiki.python.org/moin/UsingPickle

def save( file, varnames, values):
    """
    Usage: save('mydata.pysav', ['a','b','c'], [a,b,c] )
    """
    import cPickle
    f =open(file,"wb")
    super_var =dict( zip(varnames,values) )
    cPickle.dump( super_var, f )
    f.close

def restore(file):
    """
    Read data saved with save function.
    Usage: data = restore('mydata.pysav')
    a = data['a']
    b = data['b']
    c = data['c']
    """
    import cPickle
    f=open(file,"rb")
    result = cPickle.load(f)
    f.close
    return result

#------- Read ascii tables --------#
# June 11, 2013
# needed for computers that don't have access to asciidata (hotfoot)

def read_table( filename, ncols, ignore='#' ):
	"""
	Read data saved in an ascii table
	Assumes data is separated by white space
	Assumes all the data are floats
	Ignores lines that start with the ignore character / sequence
	---------------
	Usage : read_table( filename, ncols, ignore='#' )
	Returns : A dictionary with the column index as keys and the column data as lists
	"""

	# Initialize result dictionary
	result = {}
	for i in range(ncols):
		result[i] = []
	
	try : f = open( filename, 'r' )
	except:
		print 'ERROR: file not found'
		return
	
	end_of_file = False
	while not end_of_file:
		try:
			temp = f.readline()
			if temp[0] == ignore : pass  # Ignore the ignore character
			else:
				temp = temp.split()
				for i in range(ncols) : result[i].append( np.float(temp[i]) )
		except:
			end_of_file = True
	
	f.close()
	return result

