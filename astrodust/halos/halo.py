import numpy as np
from scipy.interpolate import interp1d

from .. import grainpop
from .. import constants as c

DEFAULT_ALPHA = np.linspace(5.0, 100.0, 20)

class Halo(object):
    """
    | **ATTRIBUTES**
    | htype  : abstract class containing information about the halo calculation
    | energy : float : observed energy [keV]
    | alpha  : np.ndarray : observed angle [arcsec]
    | gpop   : grainpop.GrainPop object
    | intensity : np.ndarray : fractional intensity [arcsec^-2]
    | taux   : np.ndarray : dust scattering optical depth vs energy, None (default) until calculated
    """
    def __init__(self, E0, alpha=DEFAULT_ALPHA, gpop=grainpop.make_MRN_grainpop()):
        self.htype  = None
        self.energy = E0
        self.alpha  = alpha
        self.gpop   = gpop
        self.intensity = np.zeros(np.size(alpha))
        self.taux  = None

    def ecf(self, theta, nth=500):
        """
        Compute the enclosed fraction of scattering halo surface brightness within some observation angle,
        via integral(theta,2pi*theta*halo)/tau.
            |
            | **INPUTS**
            | theta : float : Value for which to compute enclosed fraction (arcseconds)
            | nth   : int : Number of angles to use in calculation, 500 (default)
        """
        if self.htype is None:
            print 'Error: Halo has not yet beein calculated.'
            return
        interpH = interp1d(self.alpha, self.intensity)
        tharray = np.linspace(min(self.alpha), theta, nth)
        try:
            return c.intz(tharray, interpH(tharray) * 2.0*np.pi*tharray) / self.taux
        except:
            print 'Error: ECF calculation failed. Theta is likely out of bounds.'
            return
