"""
bhmie.py -- Compute Mie scattering with Bohren & Huffman (1983) algorithm, with caching
i.e. store S1 and S2 values according to E, a, and n
"""

import numpy as np

import constants as c
import cmindex as cmi

class BHmie(object):
    def __init__(self, a, E, cm):
        NA, NE = np.size(a), np.size(E)
        self.a  = np.tile(a, (NE, 1))
        self.E  = np.tile(E, (NA, 1)).T
        self.cm = cm  # complex index of refraction
        self.NA = len(a)
        self.NE = len(E)
        self.S1 = np.zeros(shape=(NA, NE, 1))
        self.S2 = np.zeros(shape=(NA, NE, 1))
        self.X  = (2.0 * np.pi * self.a) / self.E

def calculte(bhm, theta=np.array([0.0])):
    NA, NE = bhm.NA, bhm.NE
    NTH = np.size(theta)
    theta_rad = theta * c.arcs2rad

    indl90    = theta_rad < np.pi/2.0
    indg90    = theta_rad >= np.pi/2.0

    s1    = np.zeros(shape(NA, NE, NTH), dtype='complex')
    s2    = np.zeros(shape(NA, NE, NTH), dtype='complex')
    pi    = np.zeros(shape(NA, NE, NTH), dtype='complex')
    pi0   = np.zeros(shape(NA, NE, NTH), dtype='complex')
    pi1   = np.zeros(shape(NA, NE, NTH), dtype='complex') + 1.0
    tau   = np.zeros(shape(NA, NE, NTH), dtype='complex')
    amu   = np.abs(np.cos(theta_rad))

    #refrel = cm.rp(E) + 1j*cm.ip(E)
    refrel  = bhm.cm.rp(bhm.E) + 1j*bhm.cm.ip(bhm.E)

    #x      = ( 2.0 * np.pi * a*c.micron2cm ) / ( c.hc/E  )
    x      = bhm.X
    y      = x * refrel
    ymod   = np.abs(y)
    nx     = np.size(x)

    # *** Series expansion terminated after NSTOP terms
    # Logarithmic derivatives calculated from NMX on down

    xstop  = x + 4.0 * np.power(x, 0.3333) + 2.0
    test   = np.append(xstop, ymod)
    nmx    = np.max(test) + 15
    nmx    = np.int32(nmx)  # maximum number of iterations

    nstop  = xstop  # Why are we doing this?

    # *** Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    d = np.zeros(shape=(NA, NE, nmx+1), dtype='complex')
    # Original code set size to nmxx.
    # I see that the array only needs to be slightly larger than nmx

    for n in np.arange(nmx-1)+1:  # for n=1, nmx-1 do begin
        en = nmx - n + 1
        d[:, :, nmx-n]  = (en/y) - (1.0 / (d[:, :, nmx-n+1] + en/y))

    an = np.zeros(shape=(NE, NA, nmx+1), dtype='complex')
    bn = np.zeros(shape=(NE, NA, nmx+1), dtype='complex')

    # Set up for calculating Riccati-Bessel functions
    # with real argument X, calculated by upward recursion
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

    for en in np.arange(np.max(nstop)) + 1:
        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
        # *** Compute AN and BN:

        #*** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>

        psi = (2.0*en-1.0) * psi1/x - psi0
        chi = (2.0*en-1.0) * chi1/x - chi0
        xi  = psi - 1j * chi

        if en > 1:
            an1 = an[:, :, en-1]
            bn1 = bn[:, :, en-1]

        # Calculate AN and BN terms
        dslice = d[:, :, en]

        aterm  = ((dslice / refrel + en/x) * psi - psi1) /
                 ((dslice/refrel + en/x) * xi - xi1)
        an[:, :, en] = aterm

        bterm  = ((refrel * dslice + en/x) * psi - psi1) /
                 ((refrel * dslice + en/x) * xi - xi1)
        bn[:, :, en] = bterm

        # *** Augment sums for Qsca and g=<cos(theta)>

        # NOTE from LIA: In IDL version, bhmie casts double(an)
        # and double(bn).  This disgards the imaginary part.  To
        # avoid type casting errors, I use an.real and bn.real

        # Because animag and bnimag were intended to isolate the
        # real from imaginary parts, I replaced all instances of
        # double( foo * complex(0.d0,-1.d0) ) with foo.imag

        bhm.qsca  += (2.0*en +1.0) * (np.abs(aterm)**2 + np.abs(bterm)**2)
        bhm.gsca  += ((2.0*en+1.0) / (en*(en+1.0))) * \
                     (aterm.real * bterm.real + aterm.imag * bterm.imag)
        bhm.gsca  += ((en-1.0) * (en+1.0)/en) * \
                     (an1.real * an.real + an1.imag * an.imag + bn1.real * bn.real + bn1.imag * bn.imag)

        # *** Now calculate scattering intensity pattern
        #     First do angles from 0 to 90

        # LIA : Altered the two loops below so that only the indices where ang
        # < 90 are used.  Replaced (j) with [indl90]

        fn_const = (2.0 * en + 1.0) / (en * (en + 1.0))
        fn       = np.zeros(shape=(NA, NE)) + fn_const

        indl90

        pi  = pi1
        tau = en * amu * pi - (en + 1.0) * pi0

        sign = np.ones(shape=(NA, NE, NTH))
        sign[:, :, indg90] = -sign[:, :, indg90]

        s1 += fn * (an * pi + sign * bn * tau)
        s2 += fn * (an * tau + sign * bn * pi)

        pi_ext = pi1_ext
        tau_ext = en * 1.0 * pi_ext - (en + 1.0) * pi0_ext

        s1_ext = s1_ext + fn * (an * pi_ext + bn * tau_ext)
        s2_ext = s2_ext + fn * (bn * pi_ext + an * tau_ext)

        s1_back += fn * (an * pi_ext - bn * tau_ext)
        s2_back += fn * (bn * pi_ext - an * tau_ext)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1  = psi1 - 1j*chi1

        # *** Compute pi_n for next value of n
        #     For each angle J, compute pi_n+1
        #     from PI = pi_n , PI0 = pi_n-1

        pi1  = ((2.0 * en + 1.0) * amu * pi - (en + 1.0) * pi0) / en
        pi0  = pi

        pi1_ext = ((2.0 * en + 1.0) * 1.0 * pi_ext - (en + 1.0) * pi0_ext) / en
        pi0_ext = pi_ext

    # ENDFOR

    # *** Have summed sufficient terms.
    #     Now compute QSCA,QEXT,QBACK,and GSCA
    gsca = 2.0 * gsca / qsca
    qsca = (2.0 / x**2) * qsca

    qext = (4.0 / x**2) * s1_ext.real
    qback = (np.abs(s1_back)/x)**2 / np.pi
