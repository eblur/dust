
import numpy as np
import constants as c
import cmindex as cmi

## Needed for PSH model
from parse_PAH import *
from scipy.interpolate import interp1d

#-----------------------  META  -------------------------------
# A dust scattering model should contain functions that take 
# energy value, complex index of refraction object (see cmindex.py), 
# and grain sizes
#
# Qsca ( E  : scalar or np.array [keV]
#        cm : cmtype object from cmindex.py
#        a  : scalar [grain size, micron] ) :
# returns scalar or np.array [scattering efficiency, unitless]
#
# Diff ( cm : cmtype object from cmindex.py
#        theta : scalar or np.array [angle, arcsec]
#        a  : scalar [grain size, micron]
#        E  : scalar or np.array [energy, keV]
#        ** if len(E) > 1 and len(theta) > 1, then len(E) must equal len(theta)
#           returns dsigma/dOmega of values (E0,theta0), (E1,theta1) etc...
#
# Some (but not all) scattering models may also contain related extinction terms
#
# Qext ( E  : scalar or np.array [keV]
#        cm : cmtype object cmindex.py
#        a  : scalar [grain size, micron] ) :
# returns scalar or np.array [extinction efficiency, unitless]
#
# Qabs ( E  : scalar or np.array [keV]
#        cm : cmtype object cmindex.py
#        a  : scalar [grain size, micron] ) :
# returns scalar or np.array [absorption efficiency, unitless]
#
#--------------------------------------------------------------
# SCATTERING MODELS CONTAINED IN THIS FILE
#
# RGscat()
#    stype = 'RGscat'
#    Qsca( E, a=1.0, cm=cmi.CmDrude() ) : scalar or np.array [unitless]
#    Char( E=1.0, a=1.0 ) : scalar or np.array [char scattering angle, arcsec]
#    Diff( theta, E=1.0, a=1.0, cm=cmi.CmDrude() ) : scalar or np.array [diff cross-section, cm^2 ster^-1]
#
# Based on bhmie.pro (see Bohren & Huffman 1993)
# Mie()
#    stype = 'Mie'
#    getQs( a=1.0, E=1.0, cm=cmi.CmDrude(), getQ='sca', theta=None )
#    Qsca( E, a=1.0, cm=cmi.CmDrude() )
#    Qext( E, a=1.0, cm=cmi.CmDrude() )
#    Diff( theta, E=1.0, a=1.0, cm=cmi.CmDrude() )
#
# PAH( type )
#   stype = 'PAH' + type
#   type  = type
#   get_Q( E, qtype, a )
#   Qsca( E, a=0.01 )
#   Qabs( E, a=0.01 )
#   Qext( E, a=0.01 )
#--------------------------------------------------------------


class RGscat(object):
    """
    | RAYLEIGH-GANS scattering model. 
    | *see* Mauche & Gorenstein (1986), ApJ 302, 371
    | *see* Smith & Dwek (1998), ApJ, 503, 831
    
    | **ATTRIBUTES**
    | stype : string : 'RGscat'

    | **FUNCTIONS**
    | Qsca( E, a=1.0 [um], cm=cmi.CmDrude() [see cmindex.py] )
    |    *returns* scattering efficiency [unitless]
    | Char( a=1.0 [um], E=1.0 [keV] )
    |    *returns* characteristc scattering angle [arcsec keV um]
    | Diff( cm=cmi.CmDrude() [see cmindex.py], theta=10.0 [arcsec], a=1.0 [um], E=1.0 [keV] )
    |    *returns* differential scattering cross-section [cm^2 ster^-1]
    """

    stype = 'RGscat'

    def Qsca( self, E, a=1.0, cm=cmi.CmDrude() ):

        if np.size(a) != 1:
            print 'Error: Must specify only 1 value of a'
            return

        a_cm = a * c.micron2cm        # cm -- Can only be a single value
        lam  = c.hc / E               # cm -- Can be many values
        x    = 2.0 * np.pi * a_cm / lam
        mm1  = cm.rp(E) - 1 + 1j * cm.ip(E)
        return 2.0 * np.power( x, 2 ) * np.power( np.abs(mm1), 2 )

    def Char( self, a=1.0, E=1.0 ):   # Characteristic Scattering Angle(s)
        return 1.04 * 60.0 / (E * a)  # arcsec (keV/E) (um/a)

    # Can take multiple theta, but should only use one 'a' value
    # Can take multiple E, but should be same size as theta
    def Diff( self, theta, E=1.0, a=1.0, cm=cmi.CmDrude() ): # cm^2 ster^-1
        
        if np.size(a) != 1:
            print 'Error: Must specify only 1 value of a'
            return
        if np.logical_and( np.size(E) > 1,                  # If more than 1 energy is specified
                           np.size(E) != np.size(theta) ):  # theta must be of the same size
            print 'Error: If specifying > 1 energy, must have same number of values for theta'
            return

        a_cm  = a * c.micron2cm      # cm -- Can only be a single value
        lam   = c.hc / E             # cm
        x     = 2.0 * np.pi * a_cm / lam
        mm1   = cm.rp(E) + 1j * cm.ip(E) - 1
        thdep = 2./9. * np.exp( -np.power( theta/self.Char(a=a, E=E) , 2 ) / 2.0 )
        dsig  = 2.0 * np.power(a_cm,2) * np.power(x,4) * np.power( np.abs(mm1),2 )
        return dsig * thdep


## Copied from ~/code/mie/bhmie_mod.pro
## ''Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
##    to calculate scattering and absorption by a homogenous isotropic
##    sphere.''
#
class Mie(object):
    """
    | Mie scattering algorithms of Bohren & Hoffman
    | See their book: *Absorption and Scattering of Light by Small Particles*
    
    | **ATTRIBUTES**
    | stype : string : 'Mie'

    | **FUNCTIONS**
    | getQs( a=1.0 [um], E=1.0 [keV], cm=cmi.CmDrude(), getQ='sca' ['ext','back','gsca','diff'], theta=None [arcsec] )
    |     *returns* Efficiency factors depending on getQ [unitless or ster^-1]
    | Qsca( E [keV], a=1.0 [um], cm=cmi.CmDrude() )
    |     *returns* Scattering efficiency [unitless]
    | Qext( E [keV], a=1.0 [um], cm=cmi.CmDrude() )
    |     *returns* Extinction efficiency [unitless]
    | Diff( theta [arcsec], E=1.0 [keV], a=1.0 [um], cm=cmi.CmDrude() )
    |     *returns* Differential cross-section [cm^2 ster^-1]
    """
    stype = 'Mie'
    
    def getQs( self, a=1.0, E=1.0, cm=cmi.CmDrude(), getQ='sca', theta=None ):  # Takes single a and E argument
        
        if np.size(a) > 1:
            print 'Error: Can only specify one value for a'
            return
        
        indl90 = np.array([])  # Empty arrays indicate that there are no theta values set
        indg90 = np.array([])  # Do not have to check if theta != None throughout calculation
        s1     = np.array([])
        s2     = np.array([])
        pi     = np.array([])
        pi0    = np.array([])
        pi1    = np.array([])
        tau    = np.array([])
        amu    = np.array([])
        
        if theta != None:
            if np.size(E) > 1 and np.size(E) != np.size(theta):
                print 'Error: If more than one E value specified, theta must have same size as E'
                return
            
            if np.size( theta ) == 1:
                theta = np.array( [theta] )
            
            theta_rad = theta * c.arcs2rad
            amu       = np.abs( np.cos(theta_rad) )
            
            indl90    = np.where( theta_rad < np.pi/2.0 )
            indg90    = np.where( theta_rad >= np.pi/2.0 )
                        
            nang  = np.size( theta )
            s1    = np.zeros( nang, dtype='complex' )
            s2    = np.zeros( nang, dtype='complex' )
            pi    = np.zeros( nang, dtype='complex' )
            pi0   = np.zeros( nang, dtype='complex' )
            pi1   = np.zeros( nang, dtype='complex' ) + 1.0
            tau   = np.zeros( nang, dtype='complex' )
            
        refrel = cm.rp(E) + 1j*cm.ip(E)
        
        x      = ( 2.0 * np.pi * a*c.micron2cm ) / ( c.hc/E  )
        y      = x * refrel
        ymod   = np.abs(y)
        nx     = np.size( x )
        
        
        # *** Series expansion terminated after NSTOP terms
        # Logarithmic derivatives calculated from NMX on down
        
        xstop  = x + 4.0 * np.power( x, 0.3333 ) + 2.0
        test   = np.append( xstop, ymod )
        nmx    = np.max( test ) + 15
        nmx    = np.int32(nmx)
      
        nstop  = xstop
#        nmxx   = 150000
        
#        if (nmx > nmxx):
#            print 'error: nmx > nmxx=', nmxx, ' for |m|x=', ymod
        
        # *** Logarithmic derivative D(J) calculated by downward recurrence
        # beginning with initial value (0.,0.) at J=NMX
        
        d = np.zeros( shape=(nx,nmx+1), dtype='complex' )  
        dold = np.zeros( nmx+1, dtype='complex' )
        # Original code set size to nmxx.  
        # I see that the array only needs to be slightly larger than nmx
        
        for n in np.arange(nmx-1)+1:  # for n=1, nmx-1 do begin
          en = nmx - n + 1
          d[:,nmx-n]  = (en/y) - ( 1.0 / ( d[:,nmx-n+1]+en/y ) )
        
        
        # *** Riccati-Bessel functions with real argument X
        # calculated by upward recurrence
        
        psi0 = np.cos(x)
        psi1 = np.sin(x)
        chi0 = -np.sin(x)
        chi1 = np.cos(x)
        xi1  = psi1 - 1j * chi1
        
        qsca = 0.0    # scattering efficiency
        gsca = 0.0    # <cos(theta)>
        
        s1_ext = 0
        s2_ext = 0
        s1_back = 0
        s2_back = 0
        
        pi_ext  = 0
        pi0_ext = 0
        pi1_ext = 1
        tau_ext = 0
        
        p    = -1.0
        
        for n in np.arange( np.max(nstop) )+1:  # for n=1, nstop do begin
            en = n
            fn = (2.0*en+1.0)/ (en* (en+1.0))
            
            # for given N, PSI  = psi_n        CHI  = chi_n
            #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
            #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
            # Calculate psi_n and chi_n
            # *** Compute AN and BN:                                                                     
            
            #*** Store previous values of AN and BN for use
            #    in computation of g=<cos(theta)>
            if n > 1:
                an1 = an
                bn1 = bn
            
            if nx > 1:
                ig  = np.where( nstop >= n )
                
                psi    = np.zeros( nx )
                chi    = np.zeros( nx )
                
                psi[ig] = (2.0*en-1.0) * psi1[ig]/x[ig] - psi0[ig]
                chi[ig] = (2.0*en-1.0) * chi1[ig]/x[ig] - chi0[ig]
                xi      = psi - 1j * chi
                
                an = np.zeros( nx, dtype='complex' )
                bn = np.zeros( nx, dtype='complex' )
                
                an[ig] = ( d[ig,n]/refrel[ig] + en/x[ig] ) * psi[ig] - psi1[ig]
                an[ig] = an[ig] / ( ( d[ig,n]/refrel[ig] + en/x[ig] ) * xi[ig] - xi1[ig] )
                bn[ig] = ( refrel[ig]*d[ig,n] + en / x[ig] ) * psi[ig] - psi1[ig]
                bn[ig] = bn[ig] / ( ( refrel[ig]*d[ig,n] + en/x[ig] ) * xi[ig] - xi1[ig] )
            else:
                psi = (2.0*en-1.0) * psi1/x - psi0
                chi = (2.0*en-1.0) * chi1/x - chi0
                xi  = psi - 1j * chi
                
                an = ( d[0,n]/refrel + en/x ) * psi - psi1
                an = an / ( ( d[0,n]/refrel + en/x ) * xi - xi1 )
                bn = ( refrel*d[0,n] + en / x ) * psi - psi1
                bn = bn / ( ( refrel*d[0,n] + en/x ) * xi - xi1 )
            
            
            # *** Augment sums for Qsca and g=<cos(theta)>                                               
            
            # NOTE from LIA: In IDL version, bhmie casts double(an)
            # and double(bn).  This disgards the imaginary part.  To
            # avoid type casting errors, I use an.real and bn.real
            
            # Because animag and bnimag were intended to isolate the
            # real from imaginary parts, I replaced all instances of
            # double( foo * complex(0.d0,-1.d0) ) with foo.imag
            
            qsca   = qsca + ( 2.0*en +1.0 ) * ( np.power(np.abs(an),2) + np.power(np.abs(bn),2) )
            gsca   = gsca + ( ( 2.0*en+1.0 ) / ( en*(en+1.0) ) ) * ( an.real*bn.real + an.imag*bn.imag )
            
            if n > 1:
                gsca    = gsca + ( (en-1.0) * (en+1.0)/en ) * \
                    ( an1.real*an.real + an1.imag*an.imag + bn1.real*bn.real + bn1.imag*bn.imag )
            
            # *** Now calculate scattering intensity pattern
            #     First do angles from 0 to 90
            
                
            # LIA : Altered the two loops below so that only the indices where ang
            # < 90 are used.  Replaced (j) with [indl90]
            
            # Note also: If theta is specified, and np.size(E) > 1,
            # the number of E values must match the number of theta
            # values.  Cosmological halo functions will utilize this
            # Diff this way.
            
            pi  = pi1
            tau = en * amu * pi - (en + 1.0) * pi0
            
            if np.size(indl90) != 0:
                antmp = an
                bntmp = bn
                if nx > 1:
                    antmp = an[indl90]
                    bntmp = bn[indl90]  # For case where multiple E and theta are specified
                    
                s1[indl90]  = s1[indl90] + fn* (antmp*pi[indl90]+bntmp*tau[indl90])
                s2[indl90]  = s2[indl90] + fn* (antmp*tau[indl90]+bntmp*pi[indl90])
            #ENDIF
            
            pi_ext = pi1_ext
            tau_ext = en*1.0*pi_ext - (en+1.0)*pi0_ext
            
            s1_ext = s1_ext + fn* (an*pi_ext+bn*tau_ext)
            s2_ext = s2_ext + fn* (bn*pi_ext+an*tau_ext)
            
            # *** Now do angles greater than 90 using PI and TAU from
            #     angles less than 90.
            #     P=1 for N=1,3,...; P=-1 for N=2,4,...
            
            p = -p
            
            # LIA : Previous code used tau(j) from the previous loop.  How do I
            # get around this?
            
            if np.size(indg90) != 0:
                antmp = an
                bntmp = bn
                if nx > 1:
                    antmp = an[indg90]
                    bntmp = bn[indg90]  # For case where multiple E and theta are specified
                    
                s1[indg90]  = s1[indg90] + fn*p* (antmp*pi[indg90]-bntmp*tau[indg90])
                s2[indg90]  = s2[indg90] + fn*p* (bntmp*pi[indg90]-antmp*tau[indg90])
            #ENDIF
            
            s1_back = s1_back + fn*p* (an*pi_ext-bn*tau_ext)
            s2_back = s2_back + fn*p* (bn*pi_ext-an*tau_ext)
            
            psi0 = psi1
            psi1 = psi
            chi0 = chi1
            chi1 = chi
            xi1  = psi1 - 1j*chi1
            
            # *** Compute pi_n for next value of n
            #     For each angle J, compute pi_n+1
            #     from PI = pi_n , PI0 = pi_n-1
            
            pi1  = ( (2.0*en+1.0)*amu*pi- (en+1.0)*pi0 ) / en
            pi0  = pi
            
            pi1_ext = ( (2.0*en+1.0)*1.0*pi_ext - (en+1.0)*pi0_ext ) / en
            pi0_ext = pi_ext
            
            # ENDFOR
        
        # *** Have summed sufficient terms.
        #     Now compute QSCA,QEXT,QBACK,and GSCA
        gsca = 2.0 * gsca / qsca
        qsca = ( 2.0 / np.power(x,2) ) * qsca
        
        # LIA : Changed qext to use s1(theta=0) instead of s1(1).  Why did the
        # original code use s1(1)?
        
        qext = ( 4.0 / np.power(x,2) ) * s1_ext.real
        qback = np.power( np.abs(s1_back)/x, 2) / np.pi
        
        if getQ == 'sca':
            return qsca
        if getQ == 'ext':
            return qext
        if getQ == 'back':
            return qback
        if getQ == 'gsca':
            return gsca
        if getQ == 'diff':
            bad_theta = np.where( theta_rad > np.pi )  #Set to 0 values where theta > !pi
            s1[bad_theta] = 0
            s2[bad_theta] = 0
            return 0.5 * ( np.power( np.abs(s1), 2 ) + np.power( np.abs(s2), 2) ) / (np.pi * x*x)
        else:
            return 0.0
    
    
    def Qsca( self, E, a=1.0, cm=cmi.CmDrude() ):
        return self.getQs( a=a, E=E, cm=cm )
    
    def Qext( self, E, a=1.0, cm=cmi.CmDrude() ):
        return self.getQs( a=a, E=E, cm=cm, getQ='ext' )
    
    def Diff( self, theta, E=1.0, a=1.0, cm=cmi.CmDrude() ):
        
        cgeo = np.pi * np.power( a*c.micron2cm, 2 )
                           
        if np.size(a) != 1:
            print 'Error: Must specify only 1 value of a'
            return
        if np.logical_and( np.size(E) > 1, np.size(E) != np.size(theta) ):
            print 'Error: E and theta must have same size if np.size(E) > 1'
            return
        
        dQ  = self.getQs( a=a, E=E, cm=cm, getQ='diff', theta=theta )
                           
        return dQ * cgeo

class PAH( object ):
    """
    | **ATTRIBUTES**
    | type  : string : 'ion' or 'neu'
    | stype : string : 'PAH' + type
    
    | **FUNCTIONS**
    | Qsca( E, a=0.01 [um], cm=None )
    |     *returns* scattering efficiency [unitless]
    | Qabs( E, a=0.01 [um], cm=None )
    |     *returns* absorption efficiency [unitless]
    | Qext( E, a=0.01 [um], cm=None )
    |     *returns* extincton efficiency [unitless]
    """
    
    def __init__( self, type ):
        self.type  = type
        self.stype = 'PAH' + type
    
    def get_Q( self, E, qtype, a ):
        try :
            data = parse_PAH( self.type )
        except :
            print 'ERROR: Cannot find PAH type', self.type
            return
        
        try :
            qvals = np.array( data[a][qtype] )
            wavel = np.array( data[a]['w(micron)'] )
        except :
            print 'ERROR: Cannot get grain size', a, 'for', self.stype
            return
        
        # Wavelengths were listed in reverse order
        q_interp = interp1d( wavel[::-1], qvals[::-1] )
        
        E_um = ( c.hc/E ) * 1.e4   # cm to um
        return q_interp( E_um )
    
    def Qabs( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_abs', a )
    
    def Qext( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_ext', a )
    
    def Qsca( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_sca', a )





