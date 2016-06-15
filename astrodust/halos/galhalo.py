
import numpy as np

from .. import constants as c
from ..extinction import sigma_scat as ss

__all__ = ['GalHalo','path_diff','uniformISM','screenISM']

ANGLES = np.logspace(0.0, 3.5, np.int(3.5/0.05))

class GalHalo(object):
    """
    | *An htype class for storing halo properties (see halo.py)*
    |
    | **ATTRIBUTES**
    | xd  : float[0-1] : position of a dust screen
    | ismtype : string : 'Uniform' or 'Screen'
    """
    def __init__(self, xg=None, ismtype=None):
        self.xg      = xg
        self.ismtype = ismtype

#--------------- Galactic Halos --------------------

def path_diff(alpha, x):
    """
    Calculates path difference for a light ray scattered from dust at distance x=(1-d)/D, where d is distance to scatterer and D is distance to X-ray source, then observed at angle alpha.
        | **INPUTS**
        | alpha  : scalar : observation angle [arcsec]
        | x      : scalar or np.array : position of dust patch (source is at x=0, observer at x=1)
        |
        | **RETURNS**
        | alpha^2*(1-x)/(2x)
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
    Calculate the X-ray scattering intensity for dust distributed uniformly along the line of sight
        |
        | **MODIFIES**
        | halo.htype, halo.taux, halo.intensity
        |
        | **INPUTS**
        | halo : Halo object
        | NH   : float : column density [cm^-2], 1.e20 (default)
        | d2g  : float : dust-to-gass mass ratio, 0.009 (default)
        | nx   : int : number of values to use in integration, 1000 (default)
        | usepathdiff : boolean : False (default), True = use extinction due to path difference e^(-tau*path_diff)
    """
    E0    = halo.energy
    alpha = halo.alpha
    gpop  = halo.gpop
    scatm = gpop.scatm
    cmind = gpop.comp.cmind
    md    = NH * c.m_p * d2g
    nd    = gpop.sizedist.ndens(md=md)

    halo.htype = GalHalo(NH=NH, d2g=d2g, ismtype='Uniform')
    halo.taux  = ss.kappa_sca(E=E0, dist=gpop.sizedist, scatm=scatm, cm=cmind) * md

    dx    = 1.0 / nx
    xvals = np.arange(0.0, 1.0, dx) + dx

    #--- Single grain case ---

    if np.size(gpop.a) == 1:

        intensity = np.array([])
        for al in alpha:
            thscat = al / xvals  # np.size(thscat) = nx
            dsig   = ss.diff_scat(theta=thscat, a=gpop.a, E=E0, scatm=scatm, cm=cmind)

            delta_tau = 0.0
            if usepathdiff:
                print('Using path difference')
                delta_x   = path_diff(al, xvals)
                delta_tau = halo.taux * delta_x
                print(np.max(delta_x))

            itemp  = np.power(xvals, -2.0) * dsig * nd * np.exp(-delta_tau)
            intensity = np.append(intensity, c.intz(xvals, itemp))

    #--- Dust distribution case ---
    else:
        avals     = gpop.a
        intensity = np.array([])
        for al in alpha:
            thscat = al / xvals  # np.size(thscat) = nx
            iatemp    = np.array([])
            for aa in avals:
                dsig  = ss.diff_scat(E=E0, a=aa, theta=thscat, scatm=scatm, cm=cmind)
                delta_tau = 0.0
                if usepathdiff:
                    print('Using path difference')
                    delta_x   = path_diff(al, xvals)
                    delta_tau = halo.taux * delta_x
                    print max(delta_x)
                dtemp  = np.power(xvals, -2.0) * dsig * np.exp(-delta_tau)
                iatemp = np.append(iatemp, c.intz(xvals, dtemp))
            intensity = np.append(intensity, c.intz(avals, nd * iatemp))
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
    gpop  = halo.gpop
    scatm = gpop.scatm
    cmind = gpop.comp.cmindex
    md    = NH * c.m_p * d2g
    nd    = gpop.sizedist.ndens(md=md)

    halo.htype = GalHalo(xg=xg, NH=NH, d2g=d2g, ismtype='Screen')
    thscat = alpha / xg

    if np.size(gpop.a) == 1:
        dsig = ss.diff_scat(E=E0, theta=thscat, a=gpop.a, scatm=scatm, cm=cmind)
        intensity = np.power(xg, -2.0) * dsig * halo.dist.nd

    else:
        avals  = halo.dist.a
        intensity = []
        for i in range(len(alpha)):
            iatemp = np.zeros(shape=(len(avals),len(alpha)))
            for j in range(len(avals)):
                dsig    = ss.diff_scat(E=E0, theta=thscat, a=avals[j], scatm=scatm, cm=cmind)
                iatemp[j,:] = np.power(xg,-2.0) * dsig
            intensity.append(c.intz(avals, iatemp[:,i] * nd))
        intensity = np.array(intensity)

    halo.intensity = intensity * np.power(c.arcs2rad, 2)  # arcsec^-2
    halo.taux      = ss.kappa_sca(E=E0, dist=gpop.sizedist, scatm=scatm, cm=cmind) * md
    return
