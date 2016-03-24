## May 16, 2012 : Added taux to halo objects

import numpy as np
from scipy.interpolate import interp1d

from .. import constants as c
from .. import distlib
from ..extinction import sigma_scat as ss
from . import cosmology as cosmo

class Halo(object):
    """
    | **ATTRIBUTES**
    | htype : abstract class containing information about the halo calculation
    | E0    : float : observed energy [keV]
    | alpha : np.array : observed angle [arcsec]
    | dist  : distlib.DustSpectrum
    | scatm : ss.ScatModel : scattering model used
    | intensity : np.array : fractional intensity [arcsec^-2]

    | **FUNCTIONS**
    | ecf(theta, nth=500)
    |     theta : float : Value for which to compute enclosed fraction (arcseconds)
    |     nth   : int (500) : Number of angles to use in calculation
    |     *returns* the enclosed fraction for the halo surface brightness
    |     profile, via integral(theta,2pi*theta*halo)/tau.
    """
    def __init__( self, E0,
                  alpha = ss.angles(),
                  dist  = distlib.MRN_dist(),
                  scatm = ss.ScatModel() ):
        self.htype  = None
        self.energy = E0
        self.alpha  = alpha
        self.dist   = dist
        self.scatm  = scatm
        self.intensity = np.zeros( np.size(alpha) )
        self.taux  = None

    def ecf( self, theta, nth=500 ):
        if self.htype == None:
            print 'Error: Halo has not yet beein calculated.'
            return
        interpH = interp1d( self.alpha, self.intensity )
        tharray = np.linspace( min(self.alpha), theta, nth )
        try:
            return c.intz( tharray, interpH(tharray) * 2.0*np.pi*tharray ) / self.taux
        except:
            print 'Error: ECF calculation failed. Theta is likely out of bounds.'
            return
