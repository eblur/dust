"""
bhmie.py -- Compute Mie scattering with Bohren & Huffman (1983) algorithm, with caching
i.e. store S1 and S2 values according to E, a, and n
"""

import numpy as np
from scipy.special import cbrt

from astrodust import constants as c
from astrodust.distlib.composition import cmindex as cmi

class BHmie(object):
    def __init__(self, a, E, cm):
        NA, NE = np.size(a), np.size(E)
        self.a  = np.tile(a, (NE, 1))
        self.E  = np.tile(E, (NA, 1)).T
        self.cm = cm  # complex index of refraction
        self.NA = len(a)
        self.NE = len(E)
        self.X  = (2.0 * np.pi * self.a) / self.E
        self.qsca = 0.0
        self.qext = 0.0
        self.gsca = 0.0

    def calculate(self, theta=np.array([0.0])):
        NA, NE, NTH = self.NA, self.NE, len(theta)
        self.theta = theta
        self.S1  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.S2  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.s1_ext   = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.s2_ext   = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.s1_back  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.s2_back  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.pi       = np.zeros(shape=(1, NA, NE, NTH), dtype='complex') + 1.0
        self.pi_ext   = np.ones(1)
        self.psi      = np.zeros(shape=(1, NA, NE))
        self.chi      = np.zeros(shape=(1, NA, NE))
        self.xi       = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.tau = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.an  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.bn  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.D   = _calc_D(bhm)  # nmx x NA x NE
        self.qsca  = 0.0
        self.qext  = 0.0
        self.qback = 0.0
        self.gsca  = 0.0
        self.gsca_terms = np.zeros(shape=(1, NA, NE), dtype='complex')
        _calculate(self, theta)

def _calc_D(bhm):
    # *** Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    xstop  = bhm.X + 4.0 * cbrt(bhm.X) + 2.0
    test   = np.append(xstop, ymod)
    nmx    = np.max(test) + 15
    nmx    = np.int32(nmx)  # maximum number of iterations

    d = np.zeros(shape=(nmx+1, bhm.NA, bhm.NE), dtype='complex')

    for n in np.arange(nmx-1)+1:  # for n=1, nmx-1 do begin
        en = nmx - n + 1
        d[nmx-n, :, :]  = (en/y) - (1.0 / (d[nmx-n+1, :, :] + en/y))

    return d

def _calc_n(bhm, n):
    en = n + 1
    # for given N, PSI  = psi_n        CHI  = chi_n
    #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
    #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
    #              PI = pi_n ,         PI0 = pi_n-1
    if n == 0:
        psi0 = np.cos(bhm.X)  # NA x NE
        psi1 = np.sin(bhm.X)
        chi0 = -np.sin(bhm.X)
        chi1 = np.cos(bhm.X)
        pi0  = np.zeros(shape=(NA, NE, NTH), dtype='complex')
        pi   = bhm.pi[n, :, :]
        pi_ext  = bhm.pi_ext[0]
        pi0_ext = 0.0
    if n == 1:
        psi0 = np.sin(bhm.X)
        psi1 = bhm.psi[n-1, :, :]
        chi0 = np.cos(bhm.X)
        chi1 = bhm.chi[n-1, :, :]
        pi0  = bhm.pi[n-1, :, :, :]
        pi   = ((2.0*n+1.0)*amu*pi0) / n
        pi_ext  = bhm.pi_ext[n]
        pi0_ext = bhm.pi_ext[n-1]
    else:
        psi0 = bhm.psi[n-2, :, :]
        psi1 = bhm.psi[n-1, :, :]
        chi0 = bhm.chi[n-2, :, :]
        chi1 = bhm.chi[n-1, :, :]
        pi0  = bhm.pi[n-1, :, :, :]
        pi   = ((2.0*n+1.0)*amu*pi0-(n+1.0)*bhm.pi[n-2, :, :]) / n
        pi_ext  = bhm.pi_ext[n]
        pi0_ext = bhm.pi_ext[n-1]

    psi = (2.0*en-1.0) * psi1/bhm.X - psi0
    chi = (2.0*en-1.0) * chi1/bhm.X - chi0

    xi  = psi - 1j * chi
    xi1 = psi1 - 1j * chi1

    if n > 0:
        an1 = bhm.an[:, :, n-1]
        bn1 = bhm.bn[:, :, n-1]

    # Calculate AN and BN terms
    dslice = bhm.D[n, :, :]  # NA x NE
    an  = ((dslice / refrel + en/x) * psi - psi1) / \
          ((dslice/refrel + en/x) * xi - xi1)
    bn  = ((refrel * dslice + en/x) * psi - psi1) / \
          ((refrel * dslice + en/x) * xi - xi1)

    bhm.an = np.stack([bhm.an, an], 0)  # stack along axis 0
    bhm.bn = np.stack([bhm.bn, bn], 0)

    # Now calculate the S1 and S2 terms
    fn_const = (2.0 * en + 1.0) / (en * (en + 1.0))
    fn       = np.zeros(shape=(NA, NE)) + fn_const  # NA x NE

    tau = en * amu * pi - (en + 1.0) * pi0

    # Deal with angles > 90 degrees
    sign = np.ones(shape=(NA, NE, NTH))
    sign[:, :, indg90] = -sign[:, :, indg90]

    s1n = fn * (an * pi + sign * bn * tau)
    s2n = fn * (an * tau + sign * bn * pi)

    tau_ext = en * 1.0 * pi_ext - (en + 1.0) * pi0_ext

    s1_ext_n = fn * (an * pi_ext + bn * tau_ext)
    s2_ext_n = fn * (bn * pi_ext + an * tau_ext)

    s1_back_n = fn * (an * pi_ext - bn * tau_ext)
    s2_back_n = fn * (bn * pi_ext - an * tau_ext)


    # Stack everything onto the BHmie object along axis 0
    bhm.S1 = np.stack([bhm.S1, s1n], 0)
    bhm.S2 = np.stack([bhm.S2, s2n], 0)
    bhm.s1_ext = np.stack([bhm.s1_ext, s1_ext_n], 0)
    bhm.s2_ext = np.stack([bhm.s2_ext, s2_ext_n], 0)
    bhm.s1_back = np.stack([bhm.s1_back, s1_back_n], 0)
    bhm.s2_back = np.stack([bhm.s2_back, s2_back_n], 0)

    bhm.pi  = np.stack([bhm.pi, pi], 0)
    bhm.psi = np.stack([bhm.psi, psi], 0)
    bhm.chi = np.stack([bhm.chi, chi], 0)
    bhm.xi  = np.stack([bhm.xi, xi], 0)
    bhm.tau = np.stack([bhm.tau, tau], 0)

    bhm.pi_ext = np.append(bhm.pi_ext,
                           ((2.0 * en + 1.0) * 1.0 * pi_ext - (en + 1.0) * pi0_ext) / en)

    # This sum is ridiculous so I'm going to do it here
    gsca = ((2.0 * en + 1.0) / (en * (en+1.0))) * \
           (an.real * bn.real + an.imag * bn.imag)
    if n > 0:
        gsca += ((en-1.0) * (en+1.0) / en) * \
                (an1.real * an.real + an1.imag * an.imag + \
                 bn1.real * bn.real + bn1.imag * bn.imag)
    bhm.gsca_term = np.stack([bhm.gsca_term, gsca], 0)
    return

def _calculate(bhm, theta):

    NA, NE = bhm.NA, bhm.NE
    NTH = np.size(theta)
    theta_rad = theta * c.arcs2rad

    indl90    = theta_rad < np.pi/2.0
    indg90    = theta_rad >= np.pi/2.0

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

    xstop  = x + 4.0 * cbrt(x) + 2.0
    test   = np.append(xstop, ymod)
    nmx    = np.max(test) + 15
    nmx    = np.int32(nmx)  # maximum number of iterations

    nstop  = xstop  # Why are we doing this?

    # Set up for calculating Riccati-Bessel functions
    # with real argument X, calculated by upward recursion
    #
    for n in np.arange(np.max(nstop)) + 1:
        _calc_n(n)

    NIND     = bhm.S1.shape[0]
    EN       = np.tile(np.arange(NIND)+1, (NA, NE, 1)).T  # n x NA x NE
    a2_b2    = (2.0 * EN + 1.0) * (np.abs(bhm.an)**2 + np.abs(bhm.bn)**2)
    bhm.qsca = (2.0 / x**2) * np.sum(a2_b2, 0)

    bhm.gsca  = 2.0 * np.sum(bhm.gsca_terms, 0) / bhm.qsca

    bhm.qext  = (4.0 / x**2) * np.sum(bhm.s1_ext.real, 0)

    backterm  = np.sum(np.abs(bhm.s1_back), 0)
    bhm.qback = (backterm/x)**2 / np.pi
    return
