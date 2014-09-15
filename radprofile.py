
import numpy as np
import matplotlib.pyplot as plt

import errors as err
from astropy.io import ascii

## Sept 15, 2014 : Use astropy modules to read ascii data, replacing asciidata module

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
        self.surbri_err = err.prop_add(oldsb_err, value_err)
        return
    
    def plus( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb + value
        self.surbri_err = err.prop_add(oldsb_err, value_err)
        return
    
    def divide( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb / value
        self.surbri_err = err.prop_div( oldsb, value, oldsb_err, value_err )
        return
    
    def multiply( self, value, value_err=0 ):
        oldsb     = self.surbri
        oldsb_err = self.surbri_err
        self.surbri = oldsb * value
        self.surbri_err = err.prop_mult( oldsb, value, oldsb_err, value_err )
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

def get_profile( filename ):
    result = Profile()
    data = ascii.read( filename + '.txt' )
    result.rleft = data['col1'].data
    result.rright = data['col2'].data
    result.surbri = data['col3'].data # counts or flux, pix^-2
    result.surbri_err = data['col4'].data
    return result

# March 29, 2013 : logmin value contains the minimum extent of the lower errorbar;
# This is helpful for loglog plotting, when the error bar is larger than the value
#
def plot_profile( profile, logmin=None, *args, **kwargs ):
	errhi = profile.surbri_err
	errlo = profile.surbri_err
	if logmin != None:
		print 'Utilizing logmin error bars'
		ineg  = np.where( profile.surbri - profile.surbri_err <= 0.0 )[0]
		errlo[ineg] = profile.surbri[ineg] - logmin
	plt.errorbar( profile.rmid, profile.surbri, \
                  xerr=(profile.rright-profile.rleft)*0.5, \
                  yerr=[errlo,errhi], *args, **kwargs )
	return

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
    plot_profile( profile )
    plt.loglog()
    plt.xlabel( 'Pixels' )
    plt.ylabel( 'Surface Brightness [counts/pixel$^2$]' )
    plt.title( sys.argv[1] )
    plt.savefig( sys.argv[1] + '.pdf', format='pdf' )
except:
    pass


