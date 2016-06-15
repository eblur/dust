
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

from .. import constants as c
from .. import distlib

ALPHA  = np.logspace(0.0, 3.0, 30)
GPOP   = distlib.make_MRN_grainpop()

#---------------------------------------------------------------

class HaloDict(object):
    """
    A dictionary of halos where each property can be looked up with energy as a key.
    |
    | **ATTRIBUTES**
    | alpha  : np.array : Observation angles [arcsec]
    | energy : np.array : Energy values [keV]
    | index  : dict : maps energy value to integer index
    | gpop   : distlib.GrainPop
    | intensity : np.array : 2-D array with intensity values as a function of alpha and energy
    | htype  : halo type object (halo.CosmHalo or galhalo.GalHalo)
    | taux   : np.array : Scattering cross-section as a function of energy
    | total  : np.array : Total halo flux as a function of osbervation angle [e.g. phot cm^-2 s^-1 arcsec^-2]
    |
    | **PROPERTIES**
    | len    : length in E dimension
    | hsize  : length in alpha dimension
    | superE : a 2-D array with the energy values duplicated along axis 1
    | superA : a 2-D array with the observation angles duplicated along axis 0
    |
    | **CALL**
    | halodict[E] : halo intensity I_h/F_a at E (keV) [arcsec^-2]
    | halodict[emin:emax] : sum of halo intensities betwen emin (keV) and emax (keV)
    |
    | **FUNCTIONS**
    | total_halo(fluxes)
    |     fluxes : np.array : Source apparent flux as a function of energy [e.g. phot cm^-2 s^-1]
    |     *Creates / updates the self.total parameter with the sum of the halo flux from each energy bin*
    | ecf(theta, nth=500)
    |     theta : float : Value for which to compute enclosed fraction (arcseconds)
    |     nth   : int (500) : Number of angles to use in calculation
    |     *returns* the enclosed fraction for the halo surface brightness
    |     profile, as a function of energy via
    |     integral(theta,2pi*theta*halo)/total_halo_counts
    """

    def __init__(self, energy, alpha=ALPHA, gpop=GPOP):
        self.alpha  = alpha
        self.energy = energy
        self.index  = dict(zip(energy,range(len(energy))))
        self.gpop   = gpop
        self.intensity = np.zeros(shape=(len(energy), len(alpha)))

        # The following variables get defined when htype is set
        # See analytic.set_htype
        self.htype  = None
        self.taux   = None

    ## Issues with comparing flouts, try round
    ## http://stackoverflow.com/questions/23721230/float-values-as-dictionary-key
    def __getitem__(self, key, n=2):
        if isinstance(key, slice):
            halos_of_interest = (self.energy >= key.start) & (self.energy <= key.stop)
            return np.sum(self.intensity[halos_of_interest,:], axis=0)
        else:
            i = self.index[round(key,n)]
            return self.intensity[i,:]

    ## http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    def __iter__(self):
        self.count = 0
        return self
    def next(self):
        if self.count >= len(self.energy):
            raise StopIteration
        else:
            self.count += 1
            return self.intensity[count,:]

    @property
    def len(self):
        return len(self.energy)

    @property
    def hsize(self):
        return len(self.alpha)

    @property
    def superE(self):
        NE, NA = len(self.energy), len(self.alpha)
        return np.tile(self.energy.reshape(NE,1), NA)  # NE x NA

    @property
    def superA(self):
        NE = len(self.energy)
        return np.tile(self.alpha, (NE,1))  # NE x NA

    def total_halo(self, fluxes):
        NE, NA = len(self.energy), len(self.alpha)
        if len(fluxes) != NE:
            print('Error: Number of flux bins must equal the number of halo energy bins')
            return

        superflux  = np.tile(fluxes.reshape(NE,1), NA)
        result     = np.sum(superflux * self.intensity, 0)
        self.total = result
        return

        # Update -- Feb 10, 2015 to match halo.ecf
    def ecf(self, theta, nth=500):
        NE = len(self.energy)
        result = np.zeros(NE)
        taux   = self.taux
        tharray = np.linspace(min(self.alpha), theta, nth)

        if self.taux is None:
            print('ERROR: No taux is specified. Need to run halo calculation')
            return result

        for i in range(NE):
            interpH = interp1d(self.alpha, self.intensity[i,:])
            result[i] = c.intz(tharray, interpH(tharray) * 2.0*np.pi*tharray) / taux[i]

        return result

#---------------------------------------------------------------
# Supporting functions

"""
# 2016.06.10 - deprecated
def get_spectrum(filename):
    data = c.read_table( filename, 2 )
    energy, flux = np.array( data[0] ), np.array( data[1] )
    return energy, flux

def aeff( filename ):
    data = c.read_table( filename, 2 )
    energy = np.array( data[0] )
    aeff   = np.array( data[1] )
    return interp1d( energy, aeff )   # keV vs cm^2
"""
## 2015.01.29 - Add a function that will save and load halo dicts into/from fits files
def fitsify_halodict( hd, outfile, clobber=False ):
    """
    | Save a halo dictionary to a fits file, with some useful information in the header
    |
    | **INPUTS**
    | hd      : HaloDict object
    | outfile : string : output file name
    | clobber : boolean (False) : set to True to overwrite outfile if it already exists
    """
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

def read_halodict_fits(infile):
    """
    | Read in a fits file and pass on as much information as possible to a halo dict object
    |
    | **INPUTS**
    | infile : string : fits file name to read
    |
    | **RETURNS**
    | intensity values from the fits file
    """
    hdulist = fits.open(infile)
    header  = hdulist[0].header

    energy  = hdulist[1].data['energy']
    taux    = hdulist[1].data['taux']
    rad     = hdulist[2].data['rad']
    md      = header['DUSTMASS']
    rho     = header['DUSTDENS']
    powr    = header['DUSTPOWR']
    alpha   = hdulist[3].data['alpha']

    result  = HaloDict( energy, alpha=alpha, rad=rad )
    result.taux = taux
    result.dist = distlib.Dustspectrum(a=rad, p=powr, rho=rho, md=md)

    for i in np.arange(len(energy))+3:
        result.intensity[i-3,:] = hdulist[i].data['intensity']

    return result
