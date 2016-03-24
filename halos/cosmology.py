
import numpy as np

from .. import constants as c
from .. import distlib
from ..extinction import sigma_scat as ss

#-----------------------------------------------------
def zvalues( zs=4.0, z0=0.0, nz=100 ):
    """ Creates an np.array of z values between z0 and zs [unitless]
    zs=4.0, z0=0.0, nz=100 """
    dz = (zs - z0)/nz
    return np.arange( z0, zs+dz, dz )

#-----------------------------------------------------

class Cosmology(object):
    def __init__( self, h0=c.h0, m=c.omega_m, l=c.omega_l, d=c.omega_d ):
        """
        Cosmology object
        ------------------------------
        INPUTS (same as stored values)
        ------------------------------
        h0 : [km/s/Mpc] (default: constants.h0)
        m  : cosmic mass density (default: constants.omega_m)
        l  : lambda mass density (default: constants.omega_l)
        d  : cosmic dust density (default: constants.omega_d )
        """
        self.h0 = h0
        self.m  = m
        self.l  = l
        self.d  = d

class Cosmdens(object):  # h0, omega, mass density (md)
    def __init__(self, cosm=Cosmology() ):
        """
        Stores h0 and omega information from cosm
        ------------------------------
        INPUTS
        ------------------------------
        cosm : Cosmology
        ------------------------------
        STORED VALUES
        h0, d, md
        """
        self.h0    = cosm.h0
        self.omega = cosm.d
        self.md    = cosm.d * c.rho_crit * np.power( cosm.h0/c.h0, 2 )

#-----------------------------------------------------

def cosmdustspectrum( amin=0.1, amax=1.0, na=100., p=4.0, rho=3.0, cosm=Cosmology() ):
    """
    FUNCTION cosmdustspectrum( amin=0.1, amax=1.0, na=100., p=4.0, rho=3.0, cosm=Cosmology() )
    -----------------------------
    amin, amax : [microns]
    na   : int (# of dust grain sizes to use)
    p    : scalar for dust power law dn/da \propto a^-p
    rho  : grain density [g cm^-3]
    cosm : Cosmology
    -----------------------------
    RETURNS : distlib.DustSpectrum
    """
    return distlib.DustSpectrum(rad=distlib.Powerlaw(amin, amax, na=na, p=p, rho=rho), \
        md = Cosmdens( cosm=cosm ).md )

def DChi( z, zp=0.0, cosm=Cosmology(), nz=100 ):
    """
    Calculates co-moving radial distance [Gpc] from zp to z using dx = cdt/a
    z    : float : redshift
    zp   ; float (0) : starting redshift
    cosm : Cosmology
    nz   : int (100) : number of z-values to use in calculation
    """
    zvals     = zvalues( zs=z, z0=zp, nz=nz )
    integrand = c.cperh0 * ( c.h0/cosm.h0 ) / np.sqrt( cosm.m * np.power(1+zvals,3) + cosm.l )
    return c.intz( zvals, integrand ) / (1e9 * c.pc2cm ) # Gpc, in comoving coordinates

def DA( theta, z, cosm=Cosmology(), nz=100 ):
    """
    Calculates the diameter distance [Gpc] for an object of angular size
    theta and redshift z using DA = theta(radians) * DChi / (1+z)
    theta : float : angular size [arcsec]
    z     : float : redshift of object
    cosm  : Cosmology
    nz    : int (100) : number of z-values to use in DChi calculation
    """
    dchi = DChi( z, cosm=cosm, nz=nz )
    return theta * c.arcs2rad * dchi / (1+z)


def CosmTauX( z, E=1.0, dist=distlib.Powerlaw(), scatm=ss.ScatModel(), cosm=Cosmology(), nz=100 ):
    """
    FUNCTION CosmTauX( z, E=1.0, dist=distlib.Powerlaw(), scatm=ss.ScatModel(), cosm=Cosmology(), nz=100
    ---------------------------------
    INPUT
    z : redshift of source
    E : scalar or np.array [keV]
    dist  : distlib.Powerlaw or distlib.Grain
    scatm : ss.ScatModel
    cosm  : cosm.Cosmology
    ---------------------------------
    OUTPUT
    tauX : scalar or np.array [optical depth to X-ray scattering]
           = kappa( dn/da da ) cosmdens (1+z)^2 cdz/hfac
    """

    zvals     = zvalues( zs=z, nz=nz )
    md        = Cosmdens( cosm=cosm ).md
    spec      = distlib.DustSpectrum( rad=dist, md=md )

    if np.size(E) > 1:
        result = np.array([])
        for ener in E:
            Evals = ener * (1+zvals)
            kappa     = ss.Kappascat( E=Evals, scatm=scatm, dist=spec ).kappa
            hfac      = np.sqrt( cosm.m * np.power( 1+zvals, 3 ) + cosm.l )
            integrand = kappa * md * np.power( 1+zvals, 2 ) * \
                c.cperh0 * ( c.h0/cosm.h0 ) / hfac
            result    = np.append( result, c.intz( zvals, integrand ) )
    else:
        Evals     = E * (1 + zvals)
        kappa     = ss.Kappascat( E=Evals, scatm=scatm, dist=spec ).kappa
        hfac      = np.sqrt( cosm.m * np.power( 1+zvals, 3 ) + cosm.l )
        integrand = kappa * md * np.power( 1+zvals, 2 ) * c.cperh0 * ( c.h0/cosm.h0 ) / hfac
        result    = c.intz( zvals, integrand )

    return result


def CosmTauScreen( zg, E=1.0, dist=distlib.DustSpectrum(), scatm=ss.ScatModel() ):
    """
    FUNCTION CosmTauScreen( zg, E=1.0, dist=distlib.DustSpectrum(), scatm=ss.ScatModel() )
    ---------------------------------
    INPUT
    zg : redshift of screen
    E  : scalar or np.array [keV]
    dist  : distlib.Powerlaw or distlib.Grain
    scatm : ss.ScatModel
    ---------------------------------
    OUTPUT
    tauX : np.array [optical depth to X-ray scattering] for the screen
         : = kappa(Eg) * M_d
    """

    Eg  = E * (1+zg)
    kappa = ss.Kappascat( E=Eg, scatm=scatm, dist=dist ).kappa
    return dist.md * kappa
