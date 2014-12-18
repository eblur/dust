
## December 11, 2014 : Took out reliance on AEFF, which I initially took from PIMMS
## June 26, 2013 : Support code for emcee analysis

import numpy as np
import matplotlib.pyplot as plt

import radprofile as rp
import constants as c
import model_halo as MH
import dust
import sigma_scat as ss

import cPickle
from scipy.interpolate import interp1d

##-------- Supporting constants, Cyg X-3 obsid 6601 -------##

ALPHA    = np.arange( 1.0, 200.0, 1.0 ) # 0.5 arcsec resolution
AMIN     = 0.005

##-------- Supporting structure, from emcee_fit -------##

## Parse text files containing walker positions
def string_to_walker( string ):
    pos_string = string.strip().strip('[').strip(']')
    walker = np.array( [] )
    for param in pos_string.split():
        walker = np.append( walker, np.float(param) )
    return np.array( [walker] )

def read_pos( filename ):
    f = open( filename )

    first_line = f.readline()
    result     = string_to_walker( first_line )

    end_of_file = False
    while not( end_of_file ):
        try:
            next_line = f.readline()
            walker    = string_to_walker( next_line )
            result = np.concatenate( (result,walker) )
        except:
            end_of_file = True

    f.close()
    return result

def read_prob( filename ):
    prob_data = open( filename, 'r' )
    prob = []
    
    end_of_file = False
    while not end_of_file:
        try:
            newprob = prob_data.readline().strip()
            prob.append( np.float(newprob) )
        except:
            end_of_file = True
    
    prob = np.array( prob )
    prob_data.close()
    return prob

## Unpickle stuff
def eat_pickle( filename ):
    pickle_data = open( filename, 'rb' )
    data        = cPickle.load( pickle_data )
    pickle_data.close()
    return data

## Simulate the halos
def uniform_halo( filename, params, **kwargs ):
    logNH, amax, p = params
    return MH.simulate_uniform( filename, \
        NH=np.power(10.0,logNH), a0=AMIN, a1=amax, p=p, **kwargs )

def screen_halo( filename, params, **kwargs ):
    xg, logNH, amax, p = params
    return MH.simulate_screen( filename, \
        xg=xg, NH=np.power(10.0,logNH), a0=AMIN, a1=amax, p=p, **kwargs)

def sum_interp( sb1, sb2 ):
    ## Takes interp objects and sums them to create another interp object
    ## Assumes same x values for both
    if sb1.x.all() != sb2.x.all():
        print 'Error: Interp objects must have same x-axis values'
        return
    else:
        return interp1d( sb1.x, sb1.y + sb2.y )

def multiscreen_halo( specfile, params, amin=AMIN, **kwargs ):
    x1, x2, logNH1, logNH2, amax, p = params
    s1 = MH.simulate_screen( specfile, xg=x1, NH=np.power(10.0,logNH1), \
        a0=AMIN, a1=amax, p=p, **kwargs )
    s2 = MH.simulate_screen( specfile, xg=x2, NH=np.power(10.0,logNH2), \
        a0=AMIN, a1=amax, p=p, **kwargs )
    return sum_interp( s1, s2 )

def uniscreen( specfile, params, **kwargs ):
    logNHu, logNHs, a_u, a_s, p_u, p_s, x_s = params
    nhu    = np.power( 10.0, logNHu )
    UU     = MH.simulate_uniform( specfile, NH=nhu, a0=AMIN, a1=a_u, p=p_u, **kwargs )
    nhs    = np.power( 10.0, logNHs )
    SS     = MH.simulate_screen( specfile, xg=x_s, NH=nhs, a0=AMIN, a1=a_s, p=p_s, **kwargs )
    return sum_interp( UU, SS )

def red_chisq( xdata, ydata, sigma, model, nparams ):
    chi = ( ydata - model(xdata) ) / sigma
    return np.sum(chi**2) / ( len(xdata) - nparams )

def chisq( xdata, ydata, sigma, model ):
    chi = ( ydata - model(xdata) ) / sigma
    return np.sum(chi**2)

##-------- Some basic plotting stuff -------##

def plot_chains( chainfile, title=None, unit=None, opt_values=None, **kwargs ):
    
    chain = eat_pickle( chainfile )
    
    nwalkers, nsteps, ndim = chain.shape
    
    for d in range(ndim):
        plt.figure()
        for i in range(nwalkers):
            plt.plot( range(nsteps), chain[i,:,d], **kwargs )
        
        if title != None : plt.title( title[d] )
        if unit  != None : plt.ylabel( unit[d] )
        if opt_values != None :
            plt.axhline( opt_values[d], lw=3, ls='--', color='r' )
    
    return

def plot_whist( walkers, nbins, title=None, unit=None, opt_values=None, \
    histtype='step', **kwargs ):
    
    ndim = len( walkers[0] )
    
    for d in range(ndim):
        plt.figure()
        plt.hist( walkers[:,d], nbins, histtype=histtype, **kwargs )
        if title != None : plt.title( title[d] )
        if unit  != None : plt.xlabel( unit[d] )
        if opt_values != None :
            plt.axvline( opt_values[d], lw=3, ls='--', color='r' )
    
    return

def compare_walkers( w1, w2, nbins, wlabels=None, \
    title=None, unit=None, opt_values=None, histtype='stepfilled' ):
    
    ndim = len( w1[0] )
    if wlabels == None : wlabels = ['','']
    
    for d in range(ndim):
        plt.figure()
        plt.hist( w1[:,d], nbins, histtype='stepfilled', \
                color='k', alpha=0.3, label=wlabels[0] )
        plt.hist( w2[:,d], nbins, histtype='stepfilled', \
                color='b', alpha=0.3, label=wlabels[1] )
        if wlabels != None : plt.legend( loc='upper right', frameon=False )
        
        if title != None : plt.title( title[d] )
        if unit  != None : plt.xlabel( unit[d] )
        if opt_values != None :
            plt.axvline( opt_values[d], lw=3, ls='--', color='k' )
    
    return

##--------- Grab items from sample ---------##

def sample_halos( sample, isample, mscreen=False, **kwargs ):
    result = []
    if mscreen:
        for i in isample:
            x1, x2, logNH1, logNH2, amax, p = sample[i]
            NH1, NH2 = np.power(10.0,logNH1), np.power(10.0,logNH2)
            print 'x =', x1, x2, '\tNH =', NH1, NH2, '\tamax =', amax, '\tp =', p
            result.append( multiscreen_halo( x1, x2, NH1, NH2, amax=amax, p=p, **kwargs ) )
    else:
        for i in isample:
            logNH, amax, p = sample[i]
            NH = np.power(10.0,logNH)
            print 'NH =', NH, '\tamax =', amax, '\tp =', p
            result.append( uniform_halo( NH=NH, amax=amax, p=p, **kwargs ) )
    return result

def multiscreen_tau( sample, d2g=0.009, scatm=ss.makeScatmodel('RG','Drude') ):
    result = []
    for walker in sample:
        logNHu, logNHs, a_u, a_s, p_u, p_s, x_s = walker
        MDu, MDs = np.power(10.0,logNHu) * c.mp() * d2g, np.power(10.0,logNHs) * c.mp() * d2g
        da_u, da_s = (a_u-AMIN)/10.0, (a_s-AMIN)/10.0
        Udust = dust.Dustdist( rad=np.arange(AMIN,a_u+da_u,da_u), p=p_u )
        Sdust = dust.Dustdist( rad=np.arange(AMIN,a_s+da_s,da_s), p=p_s )
        Ukappa = ss.Kappascat( E=1.0, dist=dust.Dustspectrum( rad=Udust, md=MDu ), scatm=scatm ).kappa[0]
        Skappa = ss.Kappascat( E=1.0, dist=dust.Dustspectrum( rad=Sdust, md=MDs ), scatm=scatm ).kappa[0]
        result.append( Ukappa*MDu + Skappa*MDs )
    return np.array( result )

def sample_tau( sample, d2g=0.009, mscreen=False ):
    result = []
    for walker in sample:
        if mscreen: 
            x1, x2, logNH1, logNH2, amax, p = walker
            nhtot = np.power(10.0,logNH1) + np.power(10.0,logNH2)
            md    = nhtot * c.mp() * d2g
        else:
            logNH, amax, p = walker
            md = np.power(10.0,logNH) * c.mp() * d2g
        da = (amax-AMIN)/100.0
        DD = dust.Dustdist( rad=np.arange(AMIN,amax+da,da), p=p )
        DS = dust.Dustspectrum( rad=DD, md=md )
        KK = ss.Kappascat( E=1.0, dist=DS ).kappa[0]
        result.append( KK * md )
    return np.array(result)

def sample_logMD( sample, d2g=0.009, replace=False, mscreen=False ):
    if mscreen:
        nhtot = np.power(10.0,sample[:,2]) + np.power(10.0,sample[:,3])
        logmd = np.log10( nhtot * c.mp() * d2g )
        if replace:
            result = np.copy( sample )
            result[:,2] = sample[:,2] + np.log10( c.mp()*d2g )
            result[:,3] = sample[:,3] + np.log10( c.mp()*d2g )
            return result
    else:
        logmd = sample[:,0] + np.log10( c.mp()*d2g )
        if replace :
            result = np.copy( sample )
            result[:,0] = logmd
            return result
    return logmd

def sample_extinction( sample, lam, isample, \
    NA=20, d2g=0.009, scatm=ss.makeScatmodel('RG','Drude') ):
    
    energy = c.kev2lam() / lam # lam must be in cm to get keV
    logMD  = sample_logMD( sample )
    MD     = np.power( 10.0, logMD )
    
    result = []
    for i in isample:
        logNH, amax, p = sample[i]
        print 'logNH =', logNH, '\tamax =', amax, '\tp =', p
        da    = (amax-AMIN)/np.float(NA)
        dist  = dust.Dustdist( rad=np.arange(AMIN,amax+da,da), p=p )
        spec  = dust.Dustspectrum( rad=dist, md=MD[i] )
        kappa = ss.Kappascat( E=energy, dist=spec, scatm=scatm ).kappa
        result.append( 1.086 * MD[i] * kappa )

    return result

def extinction_curves( ext_list, lam, V=0.5470 ):
    # lam in micron this time
    A_V   = []
    curve = []
    for ext in ext_list:
        A_lam = interp1d( lam, ext )
        A_V.append( A_lam(V) )
        curve.append( A_lam.y / A_lam(V) )
    return np.array(A_V), curve
