
import numpy as np
from astropy.io import fits
from astropy.io import ascii
import errors as err

## April 1, 2013 : Added copy function to Profile object
## March 29, 2013 : Updated minus, plus, divide, multiply with error propagating routine (errors.py)
## March 2, 2013 : Updated Profile object with minus, plus, divide, multiply

## Part of radprofile.sh script
## Taken from CygX-3/6601/primary
## Plots a profile when used './radprofile.py rp_filename'
## where the '.txt' extension is missing from rp_filename

import os # Needed for environment variables
import sys
sys.argv

#----------------------------------------------
## The Profile object

class Profile(object):
    rleft  = 0.0
    rright = 0.0
    surbri = 0.0
    surbri_err = 0.0

    @property
    def rmid( self ):
        return 0.5 * (self.rleft + self.rright)

    @property
    def area( self ):
        return np.pi * (self.rright**2 - self.rleft**2) # pix^2

    def __getslice__( self, i,j ):
        result = Profile()
        result.rleft  = self.rleft[i:j]
        result.rright = self.rright[i:j]
        result.surbri = self.surbri[i:j]
        result.surbri_err = self.surbri_err[i:j]
        return result

    def __getitem__( self, ivals ):
        result = Profile()
        result.rleft  = self.rleft[ivals]
        result.rright = self.rright[ivals]
        result.surbri = self.surbri[ivals]
        result.surbri_err = self.surbri_err[ivals]
        return result

    def minus( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb - value
        self.surbri_err = err.prop_add( oldsb_err, value_err )
        #self.surbri_err = np.sqrt( oldsb_err**2 + value_err**2 )
        return

    def plus( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb + value
        self.surbri_err = err.prop_add( oldsb_err, value_err )
        #self.surbri_err = np.sqrt( oldsb_err**2 + value_err**2 )
        return

    def divide( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb / value
        self.surbri_err = err.prop_div( oldsb, value, oldsb_err, value_err )
        #self.surbri_err = oldsb_err*2 / value
        return

    def multiply( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb * value
        self.surbri_err = err.prop_mult( oldsb, value, oldsb_err, value_err )
        #self.surbri_err = oldsb_err*2 * value
        return

    def write( self, filename, indices='all', sci_note=False ):
        if indices == 'all':
            indices = range(len(self.rmid))

        FORMAT = "%f \t%f \t%f \t%f\n"
        if sci_note:
            FORMAT = "%e \t%e \t%e \t%e\n"

        f = open(filename, 'w')
        f.write( "# Bin_left\tBin_right\tSurbri\tSurbri_err\n" )
        for i in indices:
            f.write( FORMAT % \
            (self.rleft[i], self.rright[i], self.surbri[i], self.surbri_err[i]) )
        f.close()
        return

#----------------------------------------------
## Useful functions

def copy_profile( profile ):
	result = Profile()
	result.rleft  = np.array( profile.rleft )
	result.rright = np.array( profile.rright )
	result.surbri = np.array( profile.surbri )
	result.surbri_err = np.array( profile.surbri_err )
	return result

def get_profile_fits( filename, flux=False ):
    result = Profile()
    if flux:
        sb_key    = 'SUR_FLUX'     # phot/cm^2/s/pix^2
        sberr_key = 'SUR_FLUX_ERR'
    else:
        sb_key    = 'SUR_BRI'      # count/pix^2
        sberr_key = 'SUR_BRI_ERR'
    hdu_list = fits.open( filename )
    data     = hdu_list[1].data
    result.rleft = data['R'][:,0]
    result.rright = data['R'][:,1]
    result.surbri = data[sb_key]
    result.surbri_err = data[sberr_key]
    return result

def get_profile( filename ):
    result = Profile()
    data   = ascii.read( filename )
    keys   = data.keys()
    result.rleft  = data[keys[0]]
    result.rright = data[keys[1]]
    result.surbri = data[keys[2]]
    result.surbri_err = data[keys[3]]
    return result


def add_profile( profile1, profile2=Profile(), weight1=1.0, weight2=1.0 ):
    result = Profile()
#    if profile1.rleft != profile2.rleft or profile1.rright != profile2.rright:
#        print 'Error: Profile bins need to match up'
#        return
    result.surbri = profile1.surbri * weight1 + profile2.surbri * weight2
    result.surbri_err = np.sqrt( profile1.surbri_err**2 * weight1**2 + profile2.surbri_err**2 * weight2**2 )
    result.rleft  = profile1.rleft
    result.rright = profile1.rright
    return result

def make_bkg_profile( template, bkg_value, bkg_err=0.0 ):
    result = Profile()
    result.rleft  = template.rleft
    result.rright = template.rright
    result.surbri = np.zeros( len(template.rleft) ) + bkg_value
    result.surbri_err = np.zeros( len(template.rleft) ) + bkg_err
    return result

#----------------------------------------------
## Added Feb 5, 2013 : More useful functions

def add_bkg( profile, bkg_counts, bkg_area ):
    ## To subtract, put a - sign before bkg_counts
    bkg_surbri = bkg_counts / bkg_area
    bkg_err    = np.sqrt( bkg_counts ) / bkg_area
    sbnew = profile.surbri + bkg_surbri
    sbnew_err = np.sqrt( profile.surbri_err**2 + bkg_err**2 )
    profile.surbri = sbnew
    profile.surbri_err = sbnew_err
    return

def residual_profile( profile, model ):
    result = Profile()
    result.rleft  = np.array(profile.rleft)
    result.rright = np.array(profile.rright)
    result.surbri = profile.surbri - model.surbri
    result.surbri_err = np.sqrt( profile.surbri_err**2 + model.surbri_err**2 )
    return result

#----------------------------------------------

try:
    datafile    = sys.argv[1]
    profile     = get_profile( datafile )
except:
    pass
