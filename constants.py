
import math
import numpy as np
import scipy as sp

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


#------- Save and restor functions, similar to IDL -----
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

#---- Reports the status of a for loops --------

def loop_status( i, istop, f=0.0, time=False, init=False ):
    now     = ''
    if time:
        from time import gmtime, strftime
        now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    if init:
        print '0% done ' + now
        return 0.0
    
    percent = np.float64(i) / np.float64(istop)
    
    if percent >= 0.9 and f < 0.8:
        print '90% done ' + now
        return percent
    elif percent >= 0.8 and f < 0.7:
        print '80% done ' + now
        return percent
    elif percent >= 0.7 and f < 0.6:
        print '70% done ' + now
        return percent
    elif percent >= 0.6 and f < 0.5:
        print '60% done ' + now
        return percent
    elif percent >= 0.5 and f < 0.4:
        print '50% done ' + now
        return percent
    elif percent >= 0.4 and f < 0.3:
        print '40% done ' + now
        return percent
    elif percent >= 0.3 and f < 0.2:
        print '30% done ' + now
        return percent
    elif percent >= 0.2 and f < 0.1:
        print '20% done ' + now
        return percent
    elif percent >= 0.1 and f == 0.0:
        print '10% done ' + now
        return percent

