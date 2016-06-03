"""Calculates X-ray scattering halos in a cosmological context."""

import numpy as np
from scipy.interpolate import interp1d

from .. import constants as c
from .. import distlib
from ..extinction import sigma_scat as ss
from . import cosmology as cosmo
from .halo import Halo

class CosmHalo(object):
    """
    | *An htype class for storing halo properties*
    |
    | **ATTRIBUTES**
    | zs      : float : redshift of X-ray source
    | zg      : float : redshift of an IGM screen
    | cosm    : cosmo.Cosmology object
    | igmtype : labels the type of IGM scattering calculation : 'Uniform' or 'Screen'
    """
    def __init__( self, zs=None, zg=None, cosm=None, igmtype=None ):
        self.zs      = zs
        self.zg      = zg
        self.cosm    = cosm
        self.igmtype = igmtype

#----------------- Uniform IGM case --------------------------------

def uniformIGM( halo, zs=4.0, cosm=cosmo.Cosmology(), nz=500 ):
    """
    | Calculates the intensity of a scattering halo from intergalactic
    | dust that is uniformly distributed along the line of sight.
    |
    | **MODIFIES**
    | halo.htype, halo.dist, halo.intensity, halo.taux
    |
    | **INPUT**
    | halo : Halo object
    | zs   : float : redshift of source
    | cosm : cosmo.Cosmology
    | nz   : int : number of z-values to use in integration
    """
    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm
    print(scatm.cmodel.citation)

    halo.htype = CosmHalo(zs=zs, cosm=cosm, igmtype='Uniform')  # Stores information about this halo calc

    Dtot   = cosmo.dchi_fun(zs, cosm=cosm, nz=nz)
    zpvals = cosmo.zvalues(zs=zs-zs/nz, z0=0, nz=nz)

    DP    = np.array([])
    for zp in zpvals:
        DP = np.append(DP, cosmo.dchi_fun(zs, zp=zp, cosm=cosm))

    X     = DP/Dtot

    c_H0_cm = c.cperh0 * (c.h0 / cosm.h0)  # cm
    hfac    = np.sqrt(cosm.m * np.power(1+zpvals, 3) + cosm.l)

    Evals  = E0 * (1+zpvals)

    # Single grain case
    if np.size(halo.dist.a) == 1:
        intensity = np.array([])
        f    = 0.0
        cnt  = 0.0
        na   = np.size(alpha)
        for al in alpha:
            cnt += 1
            thscat = al / X  # np.size(thscat) = nz
            dsig   = ss.DiffScat( theta=thscat, a=halo.dist.a, E=Evals, scatm=scatm ).dsig
            itemp  = c_H0_cm/hfac * np.power( (1+zpvals)/X, 2 ) * halo.dist.nd * dsig
            intensity = np.append( intensity, c.intz( zpvals, itemp ) )
    ## Dust distribution case
    else:
        avals     = halo.dist.a
        intensity = np.array([])
        for al in alpha:
            thscat = al / X  # np.size(thscat) = nz
            iatemp    = np.array([])
            for aa in avals:
                dsig  = ss.DiffScat( theta=thscat, a=aa, E=Evals, scatm=scatm ).dsig
                dtmp  = c_H0_cm/hfac * np.power( (1+zpvals)/X, 2 ) * dsig
                iatemp = np.append( iatemp, c.intz( zpvals, dtmp ) )
            intensity = np.append( intensity, c.intz( avals, halo.dist.nd * iatemp ) )
    #----- Finally, set the halo intensity --------
    halo.intensity  = intensity * np.power( c.arcs2rad, 2 )  # arcsec^-2
    halo.taux       = cosmo.cosm_taux(zs, E=halo.energy, dist=halo.dist, scatm=halo.scatm, cosm=halo.htype.cosm)

#----------------- Infinite Screen Case --------------------------

def screenIGM( halo, zs=2.0, zg=1.0, md=1.5e-5, cosm=cosmo.Cosmology() ):
    """
    | Calculates the intensity of a scattering halo from intergalactic
    | dust that is situated in an infinitesimally thin screen somewhere
    | along the line of sight.
    |
    | **MODIFIES**
    | halo.htype, halo.dist, halo.intensity, halo.taux
    |
    | **INPUTS**
    | halo : Halo object
    | zs   : float : redshift of source
    | zg   : float : redshift of screen
    | md   : float : mass density of dust to use in screen [g cm^-2]
    | cosm : cosmo.Cosmology
    """
    if zg >= zs:
        print('%% STOP: zg must be < zs')
        return

    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm
    print(scatm.cmodel.citation)

    # Store information about this halo calculation
    halo.htype = CosmHalo(zs=zs, zg=zg, cosm=cosm, igmtype='Screen')

    X      = cosmo.dchi_fun(zs, zp=zg, cosm=cosm) / cosmo.dchi_fun(zs, cosm=cosm)  # Single value
    thscat = alpha / X                          # Scattering angle required
    Eg     = E0 * (1+zg)                        # Photon energy at the screen

    # Single grain size case
    if np.size(halo.dist.a) == 1:
        dsig = ss.DiffScat(theta=thscat, a=halo.dist.a, E=Eg, scatm=scatm).dsig
        intensity = halo.dist.nd / np.power(X, 2) * dsig
    # Distribution of grain sizes
    else:
        avals = halo.dist.a
        dsig  = np.zeros(shape=(np.size(avals), np.size(thscat)))
        for i in range(np.size(avals)):
            dsig[i,:] = ss.DiffScat(theta=thscat, a=avals[i], E=Eg, scatm=scatm).dsig
        intensity = np.array([])
        for j in range(np.size(thscat)):
            itemp = halo.dist.nd * dsig[:,j] / np.power(X,2)
            intensity = np.append(intensity, c.intz(avals, itemp))
    halo.intensity = intensity * np.power(c.arcs2rad, 2)  # arcsec^-2
    halo.taux      = cosmo.cosm_taux_screen(zg, E=halo.energy, dist=halo.dist, scatm=halo.scatm)
