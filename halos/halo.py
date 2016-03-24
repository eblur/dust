## May 16, 2012 : Added taux to halo objects

import numpy as np
from scipy.interpolate import interp1d

from .. import constants as c
from .. import distlib
from ..extinction import sigma_scat as ss
from . import cosmology as cosmo

class CosmHalo(object):
    """
    *An htype class for storing halo properties*

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

class Halo(object):
    """
    | **ATTRIBUTES**
    | htype : abstract class containing information about the halo calculation
    | E0    : float : observed energy [keV]
    | alpha : np.array : observed angle [arcsec]
    | rad   : distlib.Grain OR distlib.Dustdist : grain size distribution
    | dist  : distlib.Dustspectrum : initially NONE, stored from calculation
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
                  rad   = distlib.Grain(),
                  scatm = ss.ScatModel() ):
        self.htype  = None
        self.energy = E0
        self.alpha  = alpha
        self.rad    = rad    # Two options: distlib.Grain or distlib.Dustdist
        self.dist   = None   # distlib.Dustspectrum will be stored here when halo is calculated
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

#----------------- Uniform IGM case --------------------------------

def UniformIGM( halo, zs=4.0, cosm=cosmo.Cosmology(), nz=500 ):
    """
    Calculates the intensity of a scattering halo from intergalactic
    dust that is uniformly distributed along the line of sight.

    | **MODIFIES**
    | halo.htype, halo.dist, halo.intensity, halo.taux

    | **INPUT**
    | halo : Halo object
    | zs   : float : redshift of source
    | cosm : cosmo.Cosmology
    | nz   : int : number of z-values to use in integration
    """
    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm

    halo.htype = CosmHalo( zs=zs, cosm=cosm, igmtype='Uniform' ) # Stores information about this halo calc
    halo.dist  = distlib.Dustspectrum( rad=halo.rad, md=cosmo.Cosmdens(cosm=cosm).md )

    Dtot   = cosmo.DChi( zs, cosm=cosm, nz=nz )
    zpvals = cosmo.zvalues( zs=zs-zs/nz, z0=0, nz=nz )

    DP    = np.array([])
    for zp in zpvals:
        DP = np.append( DP, cosmo.DChi( zs, zp=zp, cosm=cosm ) )

    X     = DP/Dtot

    c_H0_cm = c.cperh0 * (c.h0 / cosm.h0)  #cm
    hfac    = np.sqrt( cosm.m * np.power( 1+zpvals, 3) + cosm.l )

    Evals  = E0 * (1+zpvals)

    ## Single grain case

    if type( halo.rad ) == distlib.Grain:

        intensity = np.array([])

        f    = 0.0
        cnt  = 0.0
        na   = np.size(alpha)
        for al in alpha:
            cnt += 1
            thscat = al / X  # np.size(thscat) = nz
            dsig   = ss.Diffscat( theta=thscat, a=halo.dist.a, E=Evals, scatm=scatm ).dsig
            itemp  = c_H0_cm/hfac * np.power( (1+zpvals)/X, 2 ) * halo.dist.nd * dsig
            intensity = np.append( intensity, c.intz( zpvals, itemp ) )

    ## Dust distribution case

    elif type( halo.rad ) == distlib.Dustdist:

        avals     = halo.dist.a
        intensity = np.array([])

        for al in alpha:
            thscat = al / X  # np.size(thscat) = nz

            iatemp    = np.array([])
            for aa in avals:
                dsig  = ss.Diffscat( theta=thscat, a=aa, E=Evals, scatm=scatm ).dsig
                dtmp  = c_H0_cm/hfac * np.power( (1+zpvals)/X, 2 ) * dsig
                iatemp = np.append( iatemp, c.intz( zpvals, dtmp ) )

            intensity = np.append( intensity, c.intz( avals, halo.dist.nd * iatemp ) )

    else:
        print '%% Must input type distlib.Grain or distlib.Dustdist'
        intensity = np.zeros( np.size(zpvals) )

    #----- Finally, set the halo intensity --------

    halo.intensity  = intensity * np.power( c.arcs2rad, 2 )  # arcsec^-2
    halo.taux       = cosmo.CosmTauX( zs, E=halo.energy, dist=halo.rad, scatm=halo.scatm, cosm=halo.htype.cosm )

#----------------- Infinite Screen Case --------------------------

def ScreenIGM( halo, zs=2.0, zg=1.0, md=1.5e-5, cosm=cosmo.Cosmology() ):
    """
    Calculates the intensity of a scattering halo from intergalactic
    dust that is situated in an infinitesimally thin screen somewhere
    along the line of sight.

    | **MODIFIES**
    | halo.htype, halo.dist, halo.intensity, halo.taux

    | **INPUTS**
    | halo : Halo object
    | zs   : float : redshift of source
    | zg   : float : redshift of screen
    | md   : float : mass density of dust to use in screen [g cm^-2]
    | cosm : cosmo.Cosmology
    """
    if zg >= zs:
        print '%% STOP: zg must be < zs'

    E0    = halo.energy
    alpha = halo.alpha
    scatm = halo.scatm

    # Store information about this halo calculation
    halo.htype = CosmHalo( zs=zs, zg=zg, cosm=cosm, igmtype='Screen' )
    halo.dist  = distlib.Dustspectrum( rad=halo.rad, md=md )

    X      = cosmo.DChi( zs, zp=zg, cosm=cosm ) / cosmo.DChi( zs, cosm=cosm )  # Single value
    thscat = alpha / X                          # Scattering angle required
    Eg     = E0 * (1+zg)                        # Photon energy at the screen

    ## Single grain size case

    if type( halo.rad ) == distlib.Grain:
        dsig = ss.Diffscat( theta=thscat, a=halo.dist.a, E=Eg, scatm=scatm ).dsig
        intensity = halo.dist.nd / np.power( X, 2 ) * dsig

    ## Distribution of grain sizes

    elif type( halo.rad ) == distlib.Dustdist:

        avals = halo.dist.a

        dsig  = np.zeros( shape=(np.size(avals), np.size(thscat)) )
        for i in range( np.size(avals) ):
            dsig[i,:] = ss.Diffscat( theta=thscat, a=avals[i], E=Eg, scatm=scatm ).dsig

        intensity = np.array([])
        for j in range( np.size(thscat) ):
            itemp = halo.dist.nd * dsig[:,j] / np.power(X,2)
            intensity = np.append( intensity, c.intz( avals, itemp ) )

    else:
        print '%% Must input type distlib.Grain or distlib.Dustdist'
        intensity = np.zeros( np.size(zpvals) )

    halo.intensity = intensity * np.power( c.arcs2rad, 2 )  # arcsec^-2
    halo.taux      = cosmo.CosmTauScreen( zg, E=halo.energy, dist=halo.dist, scatm=halo.scatm )
