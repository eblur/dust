
import dust
import sigma_scat as ss
import galhalo as gh

import numpy as np
import constants as c
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

## UPDATED June 11, 2013 : Make this file independent of asciidata

## July 17, 2012 : A library of objects and functions for simulating a
## full halo given an object's spectrum.
## See sim_cygx3.py for original code (and testing)

#---------------------------------------------------------------
# Supporting objects

class Spectrum( object ):
    def __init__( self, bin, flux ):
        self.bin  = bin     # np.array
        self.flux = flux    # np.array

    @property
    def dbin( self ):
        result = self.bin[1:] - self.bin[0:-1]
        result = np.append( result, result[-1] )
        return result    # np.array

    @property
    def dict( self ):
        return dict( zip(self.bin,self.flux) )

    def rebin( self, emin, emax, dE ):
        try:
            self.bin  = self.origbin
            self.flux = self.origflux
        except:
            self.origbin  = self.bin
            self.origflux = self.flux

        energy  = np.arange( emin, emax, dE )          # Defines the bin boundaries
        new_bin = energy[0:-1] + 0.5 * ( energy[1:] - energy[0:-1] )
        
        count    = 0
        new_flux = []
        for ee in energy[0:-1]:
            iee = np.where( np.logical_and( self.bin>=ee, self.bin<ee+dE ) )[0]
            if len(iee) == 0:
                new_flux.append( 0 )
            else:
                new_flux.append( np.sum( self.flux[iee] ) )

        self.bin  = new_bin
        self.flux = np.array( new_flux )
        return

    def restore( self ):
        try:
            self.bin  = self.origbin
            self.flux = self.origflux
        except:
            pass
        return

class HaloDict( object ):
    """
    A dictionary of halos where each property can be looked up with energy as a key.
    """
    def __init__( self, energy, alpha=np.arange(1.0,200.0,1.0), rad=dust.Grain(), scatm=ss.Scatmodel() ):
        halo_list   = []
        for ee in energy:
            halo_list.append( gh.Halo( ee, alpha=alpha, rad=rad, scatm=scatm ) )

        self.alpha  = alpha
        self.energy = energy
        self.dict   = dict( zip(energy,halo_list) )
        self.rad    = rad
        self.scatm  = scatm
        self.intensity = np.zeros( shape=( len(energy), len(alpha) ) )

	@property
	def superE( self ):
		return np.tile( alpha, (NE,1) ) # NE x NA
	
	@property
	def superA( self ):
		return np.tile( energy.reshape(NE,1), NA ) # NE x NA

    def __getitem__( self, key ):
        return self.dict[key]

    ## http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    def __iter__( self ):
        self.count = 0
        return self
    def next( self ):
        if self.count >= len( self.energy ):
            raise StopIteration
        else:
            self.count += 1
            return self[ self.energy[self.count-1] ]

    def __getslice__( self, i, j ):
        slice  = np.where( np.logical_and( self.energy>=i, self.energy<j ) )[0]
        result = HaloDict( self.energy[slice], alpha=self.alpha )
        for ee in self.energy[slice]:
            result.dict[ee] = self.dict[ee]
        return result

    def keys( self ):
        return self.dict.keys()
    def values( self ):
        return self.dict.values()

    @property
    def taux( self ):
    	return self[ self.energy[0] ].taux * self.energy[0]**2 / self.energy**2
#        result = []
#        for ee in self.energy:
#            result.append( self.dict[ee].taux[0] )
#        return np.array( result )

    @property
    def len( self ):
        return len( self.energy )
    @property
    def hsize( self ):
        return len( self.alpha )

    def total_halo( self, fluxes ):
        """
        Sums the halos from different energy bins to create a total halo profile.
        """
        NE = len(self.energy)
        if len(fluxes) != NE:
            print 'Error: Number of flux bins must equal the number of halo energy bins'
            return

        result = np.zeros( self.hsize )
        for i in range(NE):
            ee = self.energy[i]
            result += fluxes[i] * self[ee].intensity
        self.total = result
        self.bfrac = c.intz( self.alpha, result*2.0*np.pi*self.alpha ) \
            / np.sum( fluxes )  # brightness fraction

    def ecf( self, theta, nth=100 ):
        """
        Returns the enclosed fraction for the halo surface brightness
        profile, via integral(theta,2pi*theta*halo)/total_halo_counts
        -------------------------------------------------------------------------
        theta : float : Value for which to compute enclosed fraction (arcseconds)
        nth   : int (100) : Number of angles to use in calculation
        -------------------------------------------------------------------------
        WARNING -- This functions is not particularly robust.
        Interpolation method must be checked.
        -------------------------------------------------------------------------
        """
        try:
            interpH = interp1d( self.alpha, self.total )
        except:
            print 'Error: total_halo calculation has not been performed.'

        dth     = ( theta-min(self.alpha) ) / (nth-1)
        tharray = np.arange( min(self.alpha), theta + dth, dth )
        total   = c.intz( self.alpha, 2.0*np.pi*self.alpha * self.total )

        try:
            print c.intz( tharray, interpH(tharray) * 2.0*np.pi*tharray ) / total
            return c.intz( tharray, interpH(tharray) * 2.0*np.pi*tharray ) / total
        except:
            print 'Error: ECF calculation failed. Theta is likely out of bounds.'
            return


#---------------------------------------------------------------
# Supporting functions

def sum_spectra( spec1, spec2, emin, emax, dE, restore=False ):

    spec1.rebin( emin, emax, dE )
    spec2.rebin( emin, emax, dE )

    sum_bin  = spec1.bin
    sum_flux = spec1.flux + spec2.flux

    if restore:
        spec1.restore()
        spec2.restore()

    return Spectrum( sum_bin, sum_flux )

def aeff( filename ):
	data = c.read_table( filename, 2 )
	energy = np.array( data[0] )
	aeff   = np.array( data[1] )
	return interp1d( energy, aeff )   # keV vs cm^2

def bin_halo( alpha, hbright, binedges ):
    total  = []
    area   = []
    for i in range( len(binedges)-1 ):
        area.append( np.pi * ( binedges[i+1]**2 - binedges[i]**2 ) )  # pixels^2 or arcsec^2
        ia = np.where( np.logical_and( alpha>=binedges[i], alpha<binedges[i+1] ) )[0]
        if len(ia) == 0:
            total.append( 0 )
        else:
            total.append( c.intz( alpha[ia], hbright[ia] * 2*np.pi*alpha[ia] ) )
    result = np.array(total) / np.array(area)
    return result


