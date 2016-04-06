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
        self.a  = np.tile(a, (NE, 1)).T
        self.E  = np.tile(E, (NA, 1))
        self.cm = cm  # complex index of refraction
        self.NA = NA
        self.NE = NE
        self.X  = (2.0 * np.pi * self.a) / self.E
        self.qsca = 0.0
        self.qext = 0.0
        self.gsca = 0.0

    def calculate(self, theta=np.array([0.0])):
        NA, NE, NTH = self.NA, self.NE, len(theta)
        self.theta = theta
        self.S1  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.S2  = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.s1_ext   = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.s2_ext   = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.s1_back  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.s2_back  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.pi       = np.zeros(shape=(1, NA, NE, NTH), dtype='complex') + 1.0
        self.tau      = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.pi_ext   = np.array([1.0])
        self.psi      = np.zeros(shape=(1, NA, NE))
        self.chi      = np.zeros(shape=(1, NA, NE))
        self.xi       = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.an  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.bn  = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.D     = 0.0  # will be nmx x NA x NE
        self.qsca  = 0.0
        self.qext  = 0.0
        self.qback = 0.0
        self.gsca  = 0.0
        self.gsca_terms = np.zeros(shape=(1, NA, NE), dtype='complex')
        _calculate(self, theta)

def _calc_D(bhm, y, NMX):
    # *** Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX
    d = np.zeros(shape=(NMX+1, bhm.NA, bhm.NE), dtype='complex')
    for n in np.arange(NMX-1)+1:  # for n=1, nmx-1 do begin
        en = NMX - n + 1
        d[NMX-n, :, :]  = (en/y) - (1.0 / (d[NMX-n+1, :, :] + en/y))
    return d

def _calc_n(bhm, n):
    en = n + 1
    NA, NE, NTH = bhm.NA, bhm.NE, len(bhm.theta)

    theta_rad   = bhm.theta * c.arcs2rad
    amu         = np.abs(np.cos(theta_rad))
    amu_tiled   = np.tile(amu.reshape((1, 1, NTH)), (NA, NE, 1))
    indg90      = theta_rad >= np.pi/2.0

    refrel      = bhm.cm.rp(bhm.E) + 1j*bhm.cm.ip(bhm.E)

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
        pi_ext  = bhm.pi_ext[n]
        pi0_ext = 0.0
    elif n == 1:
        psi0 = np.sin(bhm.X)
        psi1 = bhm.psi[n-1, :, :]
        chi0 = np.cos(bhm.X)
        chi1 = bhm.chi[n-1, :, :]
        pi0  = bhm.pi[n-1, :, :, :]
        pi   = ((2.0*n+1.0)*amu_tiled*pi0) / n
        pi_ext  = bhm.pi_ext[n]
        pi0_ext = bhm.pi_ext[n-1]
    else:
        psi0 = bhm.psi[n-2, :, :]
        psi1 = bhm.psi[n-1, :, :]
        chi0 = bhm.chi[n-2, :, :]
        chi1 = bhm.chi[n-1, :, :]
        pi0  = bhm.pi[n-1, :, :, :]
        pi   = ((2.0*n+1.0)*amu_tiled*pi0-(n+1.0)*bhm.pi[n-2, :, :]) / n
        pi_ext  = bhm.pi_ext[n]
        pi0_ext = bhm.pi_ext[n-1]

    psi = (2.0*en-1.0) * psi1/bhm.X - psi0
    chi = (2.0*en-1.0) * chi1/bhm.X - chi0

    xi  = psi - 1j * chi
    xi1 = psi1 - 1j * chi1

    if n > 0:
        an1 = bhm.an[n-1, :, :]
        bn1 = bhm.bn[n-1, :, :]

    # Calculate AN and BN terms
    dslice = bhm.D[n, :, :]  # NA x NE
    an  = ((dslice/refrel + en/bhm.X) * psi - psi1) / \
          ((dslice/refrel + en/bhm.X) * xi - xi1)
    bn  = ((refrel * dslice + en/bhm.X) * psi - psi1) / \
          ((refrel * dslice + en/bhm.X) * xi - xi1)

    bhm.an = np.concatenate([bhm.an, an.reshape(1, NA, NE)], 0)  # stack along axis 0
    bhm.bn = np.concatenate([bhm.bn, bn.reshape(1, NA, NE)], 0)

    # Now calculate the S1 and S2 terms
    fn = (2.0 * en + 1.0) / (en * (en + 1.0))
    #fn       = np.zeros(shape=(NA, NE, NTH)) + fn_const  # NA x NE

    tau = en * amu_tiled * pi - (en + 1.0) * pi0

    # Deal with angles > 90 degrees
    sign = np.ones(shape=(NA, NE, NTH))
    sign[:, :, indg90] = -sign[:, :, indg90]

    an_tiled = np.tile(an.reshape(NA, NE, 1), (1, 1, NTH))
    bn_tiled = np.tile(bn.reshape(NA, NE, 1), (1, 1, NTH))
    s1n = fn * (an_tiled * pi + sign * bn_tiled * tau)
    s2n = fn * (an_tiled * tau + sign * bn_tiled * pi)

    tau_ext = en * 1.0 * pi_ext - (en + 1.0) * pi0_ext

    s1_ext_n = fn * (an * pi_ext + bn * tau_ext)
    s2_ext_n = fn * (bn * pi_ext + an * tau_ext)

    s1_back_n = fn * (an * pi_ext - bn * tau_ext)
    s2_back_n = fn * (bn * pi_ext - an * tau_ext)


    # Stack everything onto the BHmie object along axis 0
    bhm.S1 = np.concatenate([bhm.S1, s1n.reshape(1,NA,NE,NTH)], 0)
    bhm.S2 = np.concatenate([bhm.S2, s2n.reshape(1,NA,NE,NTH)], 0)
    bhm.s1_ext = np.concatenate([bhm.s1_ext, s1_ext_n.reshape(1,NA,NE)], 0)
    bhm.s2_ext = np.concatenate([bhm.s2_ext, s2_ext_n.reshape(1,NA,NE)], 0)
    bhm.s1_back = np.concatenate([bhm.s1_back, s1_back_n.reshape(1,NA,NE)], 0)
    bhm.s2_back = np.concatenate([bhm.s2_back, s2_back_n.reshape(1,NA,NE)], 0)

    bhm.pi  = np.concatenate([bhm.pi, pi.reshape(1,NA,NE,NTH)], 0)
    bhm.tau = np.concatenate([bhm.tau, tau.reshape(1,NA,NE,NTH)], 0)
    bhm.psi = np.concatenate([bhm.psi, psi.reshape(1,NA,NE)], 0)
    bhm.chi = np.concatenate([bhm.chi, chi.reshape(1,NA,NE)], 0)
    bhm.xi  = np.concatenate([bhm.xi, xi.reshape(1,NA,NE)], 0)

    bhm.pi_ext = np.append(bhm.pi_ext,
                           ((2.0 * en + 1.0) * 1.0 * pi_ext - (en + 1.0) * pi0_ext) / en)

    # This sum is ridiculous so I'm going to do it here
    gsca = ((2.0 * en + 1.0) / (en * (en+1.0))) * \
           (an.real * bn.real + an.imag * bn.imag)
    if n > 0:
        gsca += ((en-1.0) * (en+1.0) / en) * \
                (an1.real * an.real + an1.imag * an.imag + \
                 bn1.real * bn.real + bn1.imag * bn.imag)
    bhm.gsca_terms = np.concatenate([bhm.gsca_terms, gsca.reshape(1,NA,NE)], 0)
    return

def _calculate(bhm, theta):

    NA, NE = bhm.NA, bhm.NE
    refrel  = bhm.cm.rp(bhm.E) + 1j*bhm.cm.ip(bhm.E)

    x      = bhm.X
    y      = x * refrel
    ymod   = np.abs(y)

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
    bhm.D = _calc_D(bhm, y, nmx)
    for n in np.arange(np.max(nstop)):
        _calc_n(bhm, n)

    NIND     = bhm.S1.shape[0]
    EN       = np.tile(np.arange(NIND)+1, (NA, NE, 1)).T  # n x NA x NE
    a2_b2    = (2.0 * EN + 1.0) * (np.abs(bhm.an)**2 + np.abs(bhm.bn)**2)
    bhm.qsca = (2.0 / x**2) * np.sum(a2_b2, 0)

    bhm.gsca  = 2.0 * np.sum(bhm.gsca_terms, 0) / bhm.qsca

    bhm.qext  = (4.0 / x**2) * np.sum(bhm.s1_ext.real, 0)

    backterm  = np.sum(np.abs(bhm.s1_back), 0)
    bhm.qback = (backterm/x)**2 / np.pi
    return
