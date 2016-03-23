
## bhmie.py -- Compute Mie scattering with Bohren & Huffman (1983) algorithm, with caching
## i.e. store S1 and S2 values according to E, a, and n

import numpy as np

import constants as c
import cmindex as cmi

class BHmie(object):
    def __init__(self, a, E, cm):
        NA, NE = np.size(a), np.size(E)
        self.a  = np.tile(a, (NE,1))
        self.E  = np.tile(E, (NA,1)).T
        self.cm = cm # complex index of refraction
        self.NA = len(a)
        self.NE = len(E)
        self.S1 = np.zeros(shape=(NA,NE,1))
        self.S2 = np.zeros(shape=(NA,NE,1))
        self.X  = (2.0 * np.pi * self.a) / self.E

def calc(bhm):
    indl90 = np.array([])  # Empty arrays indicate that there are no theta values set
    indg90 = np.array([])  # Do not have to check if theta != None throughout calculation
    s1     = np.array([])
    s2     = np.array([])
    pi     = np.array([])
    pi0    = np.array([])
    pi1    = np.array([])
    tau    = np.array([])
    amu    = np.array([])

    #refrel = cm.rp(E) + 1j*cm.ip(E)
    refrel  = cm.rp(bhm.E) + 1j*cm.ip(bhm.E)

    #x      = ( 2.0 * np.pi * a*c.micron2cm ) / ( c.hc/E  )
    x      = bhm.X
    y      = x * refrel
    ymod   = np.abs(y)
    nx     = np.size( x )


    # *** Series expansion terminated after NSTOP terms
    # Logarithmic derivatives calculated from NMX on down

    xstop  = x + 4.0 * np.power( x, 0.3333 ) + 2.0
    test   = np.append( xstop, ymod )
    nmx    = np.max( test ) + 15
    nmx    = np.int32(nmx)

    nstop  = xstop # Why are we doing this?


    # *** Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    d = np.zeros( shape=(NA,NE,nmx+1), dtype='complex' )
    # Original code set size to nmxx.
    # I see that the array only needs to be slightly larger than nmx

    for n in np.arange(nmx-1)+1:  # for n=1, nmx-1 do begin
        en = nmx - n + 1
        d[:,:,nmx-n]  = (en/y) - ( 1.0 / ( d[:,:,nmx-n+1]+en/y ) )

    ## Set up for calculating Riccati-Bessel functions
    ## with real argument X, calculated by upward recursion
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)
    xi1  = psi1 - 1j * chi1

    qsca = 0.0
    gsca = 0.0

    s1_ext = 0.0
    s2_ext = 0.0
    s1_back = 0.0
    s2_back = 0.0

    pi_ext  = 0.0
    pi0_ext = 0.0
    pi1_ext = 1.0
    tau_ext = 0.0

    p = -1.0

    for en in np.arange(np.max(nstop)) + 1:
        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
        # *** Compute AN and BN:

        #*** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
        if en > 1:
            an1 = an
            bn1 = bn

        if nx > 1:
            ig  = (nstop >= en)

            psi    = np.zeros( shape=(NA,NE,nmx) )
            chi    = np.zeros( shape=(NA,NE,nmx) )

            psi[ig] = (2.0*en-1.0) * psi1[ig]/x[ig] - psi0[ig]
            chi[ig] = (2.0*en-1.0) * chi1[ig]/x[ig] - chi0[ig]
            xi      = psi - 1j * chi

            an = np.zeros( shape=(NA,NE,nmx), dtype='complex' )
            bn = np.zeros( shape=(NA,NE,nmx), dtype='complex' )

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
