
import os
import numpy as np
from scipy.interpolate import interp1d

from ... import constants as c

__all__ = ['CmDrude', 'CmGraphite', 'CmSilicate']

#------------- Index of Refraction object comes in handy --

#class CM(object):       # Complex index of refraction
#    def __init__( self, rp=1.0, ip=0.0 ):
#        self.rp = rp    # real part
#        self.ip = ip    # imaginary part

#-------------- Complex index of refraction calcs ---------

# ALL CM OBJECTS CONTAIN
#  cmtype : string ('Drude', 'Graphite', or 'Silicate')
#  rp     : either a function or scipy.interp1d object that is callable
#         : rp(E) where E is in [keV]
#  ip     : same as above, ip(E) where E is in [keV]


def find_cmfile( name ):
    root_path = os.path.dirname(__file__).rstrip('composition')
    data_path = root_path + 'tables/'
    return data_path + name

class CmDrude(object):
    """
    | **ATTRIBUTES**
    | cmtype : 'Drude'
    | rho    : grain density [g cm^-3]
    | citation : A string containing citation to original work
    |
    | ** FUNCTIONS**
    | rp(E)  : real part of complex index of refraction [E in keV]
    | ip(E)  : imaginary part of complex index of refraction [always 0.0]
    """
    def __init__(self, rho=3.0):  # Returns a CM using the Drude approximation
        self.cmtype = 'Drude'
        self.rho    = rho
        self.citation = "Using the Drude approximation.\nBohren, C. F. & Huffman, D. R., 1983, Absorption and Scattering of Light by Small Particles (New York: Wiley)"

    def rp(self, E):
        mm1 = self.rho / (2.0*c.m_p) * c.r_e/(2.0*np.pi) * np.power(c.hc/E, 2)
        return mm1+1

    def ip(self, E):
        if np.size(E) > 1:
            return np.zeros(np.size(E))
        else:
            return 0.0

class CmGraphite(object):
    """
    | **ATTRIBUTES**
    | cmtype : 'Graphite'
    | size   : 'big' or 'small'
    | orient : 'perp' or 'para'
    | citation : A string containing citation to original work
    | rp(E)  : scipy.interp1d object
    | ip(E)  : scipy.interp1d object [E in keV]
    """
    def __init__(self, size='big', orient='perp'):
        # size : string ('big' or 'small')
        #      : 'big' gives results for 0.1 um sized graphite grains at 20 K [Draine (2003)]
        #      : 'small' gives results for 0.01 um sized grains at 20 K
        # orient : string ('perp' or 'para')
        #        : 'perp' gives results for E-field perpendicular to c-axis
        #        : 'para' gives results for E-field parallel to c-axis
        #
        self.cmtype = 'Graphite'
        self.size   = size
        self.orient = orient
        self.citation = "Using optical constants for graphite,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"

        D03file = find_cmfile('CM_D03.pysav') # look up file
        D03vals = c.restore(D03file) # read in index values

        if size == 'big':
            if orient == 'perp':
                lamvals = D03vals['Cpe_010_lam']
                revals  = D03vals['Cpe_010_re']
                imvals  = D03vals['Cpe_010_im']

            if orient == 'para':
                lamvals = D03vals['Cpa_010_lam']
                revals  = D03vals['Cpa_010_re']
                imvals  = D03vals['Cpa_010_im']

        if size == 'small':

            if orient == 'perp':
                lamvals = D03vals['Cpe_001_lam']
                revals  = D03vals['Cpe_001_re']
                imvals  = D03vals['Cpe_001_im']

            if orient == 'para':
                lamvals = D03vals['Cpa_001_lam']
                revals  = D03vals['Cpa_001_re']
                imvals  = D03vals['Cpa_001_im']

        lamEvals = c.hc / c.micron2cm / lamvals # keV
        self.rp  = interp1d( lamEvals, revals )
        self.ip  = interp1d( lamEvals, imvals )

class CmSilicate(object):
    """
    | **ATTRIBUTES**
    | cmtype : 'Silicate'
    | citation : A string containing citation to the original work
    | rp(E)  : scipy.interp1d object
    | ip(E)  : scipy.interp1d object [E in keV]
    """
    def __init__( self ):
        self.cmtype = 'Silicate'
        self.citation = "Using optical constants for astrosilicate,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"


        D03file = find_cmfile('CM_D03.pysav')
        D03vals = c.restore(D03file)      # look up file

        lamvals = D03vals['Sil_lam']
        revals  = D03vals['Sil_re']
        imvals  = D03vals['Sil_im']

        lamEvals = c.hc / c.micron2cm / lamvals # keV
        self.rp  = interp1d( lamEvals, revals )
        self.ip  = interp1d( lamEvals, imvals )

#------------- A quick way to grab a single CM ------------

def getCM( E, model=CmDrude() ):
    """
    | **INPUTS**
    | E     : scalar or np.array [keV]
    | model : any Cm-type object
    |
    | **RETURNS**
    | Complex index of refraction : scalar or np.array of dtype='complex'
    """
    return model.rp(E) + 1j * model.ip(E)
