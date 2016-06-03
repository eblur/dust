
import numpy as np
from scipy.interpolate import interp1d

from .. import constants as c
from .. import distlib
from ..extinction import sigma_scat as ss
from . import cosmology as cosmo
from .halo import Halo

__all__ = ['GalHalo','Ihalo','path_diff','uniformISM','screenISM']

ANGLES = np.logspace(0.0, 3.5, np.int(3.5/0.05))

class GalHalo(object):
    """
    | *An htype class for storing halo properties (see halo.py)*
    |
    | **ATTRIBUTES**
    | NH  : float : hydrogen column density [cm^-2]
    | d2g : float : dust-to-gas mass ratio
    | xd  : float[0-1] : position of a dust screen
    | ismtype : string : 'Uniform' or 'Screen'
    """
    def __init__( self, NH=None, d2g=None, xg=None, ismtype=None ):
        self.NH      = NH
        self.d2g     = d2g
        self.xg      = xg
        self.ismtype = ismtype

class Ihalo(object):
    """
    | A self-similar halo object [i(theta)], azimuthally symmetric, interpolatable
    |
    | **ATTRIBUTES**
    | theta : np.array : theta values used to derive the object [arcsec]
    | itemp : np.array : values with respective theta [cm^2 arcsec^-2]
    | rad   : float : Grain size used to derive the object [um]
    | ener  : float : Photon energy used to derive the object [keV]
    | scatm : ss.ScatModel : Scattering model used to derive the object
    |
    | **CALL**
    | ihalo( theta ) : [cm^2 arcsec^-2]
    """
    def __init__( self, theta=ANGLES, \
                      scatm=ss.ScatModel(), \
                      rad=0.1, ener=1.0, nx=1000 ):
        # Theta automatically sampled in log space.
        # If I don't do it this way, linear interpolation easily fails
        # for small angles (between theta[0] and theta[1]).  Since
        # most of my plots are in log-space, it makes more sense to
        # sample logarithmically.

        if np.size(theta) < 2:
            print('Error: Must give more than one theta value')
            self.theta = None
            self.rad   = None
            self.ener  = None
            self.scatm = None

        self.theta = theta
        self.rad   = rad
        self.ener  = ener
        self.scatm = scatm

        dxi = 1.0 / np.float(nx)
        xi  = np.arange( 0, 1.0, dxi ) + dxi
        itemp = np.array([])

        for th in self.theta:
            thscat = th / xi
            dsig = ss.DiffScat( theta=thscat, scatm=self.scatm, E=self.ener, a=self.rad ).dsig
            itemp = np.append( itemp, \
                                   c.intz( xi, dsig/(xi**2) ) )

        self.itemp = itemp * c.arcs2rad**2


    def ihalo( self, theta ):

        if self.theta == None:
            print('Error: Empty ihalo object')
            return

        min_th = np.min(self.theta)
        max_th = np.max(self.theta)

        if np.min(theta) < min_th:
            print('Note: Lower bounds of interpolation exceeded.')
        if np.max(theta) > max_th:
            print('Note: Upper bounds of interpolation exceeded.')

        just_right = np.where( np.logical_and( theta >= min_th, theta <= max_th ) )

        halo_interp = interp1d( self.theta, self.itemp )
        result = np.zeros( np.size(theta) )
        result[just_right] = halo_interp( theta[just_right] )

        return result

#--------------- Galactic Halos --------------------

def path_diff(alpha, x):
    """
    | **INPUTS**
    | alpha  : scalar : observation angle [arcsec]
    | x      : scalar or np.array : position of dust patch (source is at x=0, observer at x=1)
    |
    | **RETURNS**
    | path difference associated with a particular alpha and x : alpha^2*(1-x)/(2x)
    """

    if np.size(alpha) > 1:
        print('Error: np.size(alpha) cannot be greater than one.')
        return
    if np.max(x) > 1.0 or np.min(x) < 0:
        print('Error: x must be between 0 and 1')
        return
    alpha_rad = alpha * c.arcs2rad
    return alpha_rad**2 * (1-x) / (2*x)

# May 16, 2012: Added e^-kappa_x \delta x to the integral
def uniformISM(halo, NH=1.0e20, d2g=0.009, nx=1000, usepathdiff=False):
    """
    | Calculate the X-ray scattering intensity for dust distributed
    | uniformly along the line of sight
    |
    | **MODIFIES**
    | halo.htype, halo.dist, halo.taux, halo.intensity
    |
    | **INPUTS**
    | halo : Halo object
    | NH   : float : column density [cm^-2]
    | d2g  : float : dust-to-gass mass ratio
    | nx   : int : number of values to use in integration
    | usepathdiff : boolean : True = use extinction due to path difference e^(-tau*path_diff)
    """
    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm
    md    = NH * c.m_p * d2g

    halo.htype = GalHalo(NH=NH, d2g=d2g, ismtype='Uniform')
    halo.dist  = distlib.MRN_dist(md=md)
    halo.taux  = ss.KappaScat(E=halo.energy, scatm=halo.scatm, dist=halo.dist).kappa * md

    dx    = 1.0 / nx
    xvals = np.arange(0.0, 1.0, dx) + dx

    #--- Single grain case ---

    if np.size(halo.dist.a) == 1:

        intensity = np.array([])
        for al in alpha:
            thscat = al / xvals  # np.size(thscat) = nx
            dsig   = ss.DiffScat(theta=thscat, a=halo.dist.a, E=E0, scatm=scatm).dsig

            delta_tau = 0.0
            if usepathdiff:
                print('Using path difference')
                delta_x   = path_diff(al, xvals)
                delta_tau = halo.taux * delta_x
                print(np.max(delta_x))

            itemp  = np.power(xvals, -2.0) * dsig * halo.dist.nd * np.exp(-delta_tau)
            intensity = np.append(intensity, c.intz(xvals, itemp))

    #--- Dust distribution case ---
    else:
        avals     = halo.dist.a
        intensity = np.array([])
        for al in alpha:
            thscat = al / xvals  # np.size(thscat) = nx
            iatemp    = np.array([])
            for aa in avals:
                dsig  = ss.DiffScat(theta=thscat, a=aa, E=E0, scatm=scatm).dsig
                delta_tau = 0.0
                if usepathdiff:
                    print('Using path difference')
                    delta_x   = path_diff(al, xvals)
                    delta_tau = halo.taux * delta_x
                    print max(delta_x)
                dtemp  = np.power(xvals, -2.0) * dsig * np.exp(-delta_tau)
                iatemp = np.append(iatemp, c.intz(xvals, dtemp))
            intensity = np.append(intensity, c.intz(avals, halo.dist.nd * iatemp))
    # Set the halo intensity
    halo.intensity  = intensity * np.power(c.arcs2rad, 2)  # arcsec^-2
    # halo.taux set at beginning of function so it could be called for later use
    return

def screenISM(halo, xg=0.5, NH=1.0e20, d2g=0.009):
    """
    | Calculate the X-ray scattering intensity for dust in an
    | infinitesimally thin wall somewhere on the line of sight.
    |
    | **MODIFIES**
    | halo.htype, halo.dist, halo.taux, halo.intensity
    |
    | **INPUTS**
    | halo : Halo object
    | xg   : float : distance FROM source / distance between source and observer
    | NH   : float : column density [cm^-2]
    | d2g  : float : dust-to-gass mass ratio
    """
    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm
    md    = NH * c.m_p * d2g

    halo.htype = GalHalo(xg=xg, NH=NH, d2g=d2g, ismtype='Screen')
    halo.dist  = distlib.MRN_dist(md=md)

    thscat = alpha / xg

    if np.size(halo.dist.a) == 1:
        dsig = ss.DiffScat(theta=thscat, a=halo.dist.a, E=E0, scatm=scatm).dsig
        intensity = np.power(xg, -2.0) * dsig * halo.dist.nd

    else:
        avals  = halo.dist.a
        intensity = []
        for i in range(len(alpha)):
            iatemp = np.zeros(shape=(len(avals),len(alpha)))
            for j in range(len(avals)):
                dsig    = ss.DiffScat(theta=thscat, a=avals[j], E=E0, scatm=scatm).dsig
                iatemp[j,:] = np.power(xg,-2.0) * dsig
            intensity.append(c.intz(avals, iatemp[:,i] * halo.dist.nd))
        intensity = np.array(intensity)

    halo.intensity = intensity * np.power(c.arcs2rad, 2)  # arcsec^-2
    halo.taux      = ss.KappaScat(E=halo.energy, scatm=halo.scatm, dist=halo.dist).kappa * md
    return
