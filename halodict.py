
import dust
import sigma_scat as ss
import galhalo as gh

import numpy as np
import constants as c
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from astropy.io import fits

## UPDATED July 10, 2013 : Rewrote ecf method in HaloDict object
## UPDATED June 11, 2013 : Make this file independent of asciidata

## July 17, 2012 : A library of objects and functions for simulating a
## full halo given an object's spectrum.
## See sim_cygx3.py for original code (and testing)

#---------------------------------------------------------------

class HaloDict( object ):
    """
    A dictionary of halos where each property can be looked up with energy as a key.
    """
    
    def __init__( self, energy, alpha=np.power(10.0, np.arange(0.0,3.01,0.05)), \
        rad=dust.Grain(), scatm=ss.Scatmodel() ):
    
        self.alpha  = alpha
        self.energy = energy
        self.index  = dict( zip(energy,range(len(energy))) )
        self.rad    = rad
        self.scatm  = scatm
        self.intensity = np.zeros( shape=( len(energy), len(alpha) ) )
        
        # The following variables get defined when htype is set
        # See analytic.set_htype
        self.htype  = None
        self.dist   = None
        self.taux   = None
    
    ## Issues with comparing flouts, try round
    ## http://stackoverflow.com/questions/23721230/float-values-as-dictionary-key
    def __getitem__( self, key, n=2 ):
        i = self.index[round(key,n)]
        return self.intensity[i,:]
    
    ## http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    def __iter__( self ):
        self.count = 0
        return self
    def next( self ):
        if self.count >= len( self.energy ):
            raise StopIteration
        else:
            self.count += 1
            return self.intensity[count,:]
    
    def __getslice__( self, i, j ):
        slice  = np.where( np.logical_and( self.energy>=i, self.energy<j ) )[0]
        return sum( self.intensity[slice,:], 0 )
    
    @property
    def len( self ):
        return len( self.energy )
    
    @property
    def hsize( self ):
        return len( self.alpha )
    
    @property
    def superE( self ):
        NE, NA = len(self.energy), len(self.alpha)
        return np.tile( self.energy.reshape(NE,1), NA ) # NE x NA
    
    @property
    def superA( self ):
        NE, NA = len(self.energy), len(self.alpha)
        return np.tile( self.alpha, (NE,1) ) # NE x NA
    
    def total_halo( self, fluxes ):
        """
        Sums the halos from different energy bins to create a total halo profile.
        Creates / updates the self.total parameter
        """
        NE, NA = len(self.energy), len(self.alpha)
        if len(fluxes) != NE:
            print 'Error: Number of flux bins must equal the number of halo energy bins'
            return
        
        superflux  = np.tile( fluxes.reshape(NE,1), NA )
        result     = np.sum( superflux * self.intensity, 0 )
        self.total = result
        return

        # Rewrite -- July 10, 2013
    def ecf( self, theta, nth=100 ):
        """
        Returns the enclosed fraction for the halo surface brightness
        profile, as a function of energy via 
        integral(theta,2pi*theta*halo)/total_halo_counts
        -------------------------------------------------------------------------
        theta : float : Value for which to compute enclosed fraction (arcseconds)
        nth   : int (100) : Number of angles to use in calculation
        -------------------------------------------------------------------------
        WARNING -- This functions is not particularly robust.
        Interpolation method must be checked.
        -------------------------------------------------------------------------
        """
        NE, NA = len(self.energy), len(self.alpha)
        result = np.zeros( NE ) # NE x NA
        total  = self.taux
        
        dth     = ( theta-min(self.alpha) ) / (nth-1)
        tharray = np.arange( min(self.alpha), theta + dth, dth )
        
        if self.taux == None:
            print 'ERROR: No taux is specified. Need to run halo calculation'
            return result
        
        for i in range(NE):
            interpH = interp1d( self.alpha, self.intensity[i,:] )
            result[i] = c.intz( tharray, interpH(tharray) * 2.0*np.pi*tharray ) / total[i]
        
        return result

#---------------------------------------------------------------
# Supporting functions

def get_spectrum( filename ):
    data = c.read_table( filename, 2 )
    energy, flux = np.array( data[0] ), np.array( data[1] )
    return energy, flux

def aeff( filename ):
    data = c.read_table( filename, 2 )
    energy = np.array( data[0] )
    aeff   = np.array( data[1] )
    return interp1d( energy, aeff )   # keV vs cm^2

## 2015.01.29 - Add a function that will save and load halo dicts into/from fits files

def fitsify_halodict( hd, outfile, clobber=False ):
    # Set up the header
    prihdr = fits.Header()
    prihdr['COMMENT'] = "This is a fits file containing halo dictionary information"
    prihdr['SCATM']   = hd.scatm.stype
    prihdr['HTYPE']   = hd.htype.ismtype
    prihdr['DUSTMASS'] = hd.dist.md
    prihdr['DUSTDENS'] = hd.rad.rho
    prihdr['DUSTPOWR'] = hd.rad.p
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    # Create a block that contains the energy and taux info
    tbhdu = fits.BinTableHDU.from_columns( \
        [fits.Column(name='energy', format='1E', array=hd.energy), \
         fits.Column(name='taux', format='1E', array=hd.taux)] )
    
    dust_hdu = fits.BinTableHDU.from_columns( \
        [fits.Column(name='rad', format='1E', array=hd.rad.a)] )
            
    thdulist = fits.HDUList([prihdu, tbhdu, dust_hdu])
    
    for EE in hd.energy:
        tbhdu = fits.BinTableHDU.from_columns( \
            [fits.Column(name='alpha', format='1E', array=hd.alpha), \
             fits.Column(name='intensity', format='1E', array=hd[EE])] )
        thdulist.append(tbhdu)
    
    thdulist.writeto(outfile, clobber=clobber)
    return

def read_halodict_fits( infile ):
    hdulist = fits.open(infile)
    header  = hdulist[0].header
    
    energy  = hdulist[1].data['energy']
    taux    = hdulist[1].data['taux']
    rad     = hdulist[2].data['rad']
    md      = header['DUSTMASS']
    rho     = header['DUSTDENS']
    powr    = header['DUSTPOWR']
    alpha   = hdulist[3].data['alpha']
    
    dd      = dust.Dustdist( rad=rad, p=powr, rho=rho )
    result  = HaloDict( energy, alpha=alpha, rad=dd )
    result.taux = taux
    result.dist = dust.Dustspectrum( rad=dd, md=md )
    
    for i in np.arange(len(energy))+3:
        result.intensity[i-3,:] = hdulist[i].data['intensity']
    
    return result
    
    



