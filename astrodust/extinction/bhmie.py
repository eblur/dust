"""
bhmie.py -- Compute Mie scattering with Bohren & Huffman (1983) algorithm,
  with some caching (store S1 and S2 values according to E, a, and n)
"""

import numpy as np
from scipy.special import cbrt

from astrodust import constants as c

__all__ = ['BHmie']

class BHmie(object):
    def __init__(self, a, E, cm):
        NA, NE = np.size(a), np.size(E)
        self.a  = np.tile(a, (NE, 1)).T
        self.E  = np.tile(E, (NA, 1))
        # a quick check
        assert np.sum(self.a[:,0] - a) == 0
        assert np.sum(self.E[0,:] - E) == 0

        self.cm = cm  # complex index of refraction
        self.NA = NA
        self.NE = NE
        self.X  = (2.0 * np.pi * self.a) / self.E
        self.Qsca = 0.0
        self.Qext = 0.0
        self.gsca = 0.0

    def calculate(self, theta=np.array([0.0])):
        NA, NE, NTH = self.NA, self.NE, np.size(theta)
        # Angular stuff
        self.theta = theta
        self.NTH   = NTH
        # Additive stuff
        self.S1      = np.zeros(shape=(NA, NE, NTH), dtype='complex')
        self.S2      = np.zeros(shape=(NA, NE, NTH), dtype='complex')
        self.s1_ext  = np.zeros(shape=(NA, NE), dtype='complex')
        self.s2_ext  = np.zeros(shape=(NA, NE), dtype='complex')
        self.s1_back = np.zeros(shape=(NA, NE), dtype='complex')
        self.s2_back = np.zeros(shape=(NA, NE), dtype='complex')
        # Stuff that needs to be referenced to calculate new terms
        self.pi       = np.zeros(shape=(1, NA, NE, NTH), dtype='complex') + 1.0
        self.tau      = np.zeros(shape=(1, NA, NE, NTH), dtype='complex')
        self.pi_ext   = np.array([1.0])
        self.psi      = np.zeros(shape=(1, NA, NE))
        self.chi      = np.zeros(shape=(1, NA, NE))
        self.xi       = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.an       = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.bn       = np.zeros(shape=(1, NA, NE), dtype='complex')
        self.gsca_terms = np.zeros(shape=(1, NA, NE), dtype='complex')
        # Scalars
        self.D     = 0.0  # will be nmx x NA x NE
        self.Qsca  = 0.0
        self.Qext  = 0.0
        self.Qback = 0.0
        self.gsca  = 0.0
        _calculate(self, theta)
        self.Diff = _calc_diff(self)

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
    indg90      = theta_rad >= np.pi/2.0

    amu         = np.abs(np.cos(theta_rad))
    amu_tiled  = np.repeat(np.repeat(amu.reshape((1,1,NTH)), NA, axis=0),
                           NE, axis=1)
    # a quick test
    assert np.sum(amu_tiled[0,0,:] - amu) == 0

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

    an_tiled = np.repeat(an.reshape(NA, NE, 1), NTH, axis=2)
    bn_tiled = np.repeat(bn.reshape(NA, NE, 1), NTH, axis=2)
    # make sure its doing what I want
    assert np.sum(an_tiled[:,:,0] - an) == 0
    assert np.sum(bn_tiled[:,:,0] - bn) == 0

    s1n = fn * (an_tiled * pi + sign * bn_tiled * tau)
    s2n = fn * (an_tiled * tau + sign * bn_tiled * pi)

    tau_ext = en * 1.0 * pi_ext - (en + 1.0) * pi0_ext

    s1_ext_n = fn * (an * pi_ext + bn * tau_ext)
    s2_ext_n = fn * (bn * pi_ext + an * tau_ext)

    s1_back_n = fn * (an * pi_ext - bn * tau_ext)
    s2_back_n = fn * (bn * pi_ext - an * tau_ext)

    # The following are just additive so we don't need to stor
    bhm.S1      += s1n
    bhm.S2      += s2n
    bhm.s1_ext  += s1_ext_n
    bhm.s2_ext  += s2_ext_n
    bhm.s1_back += s1_back_n
    bhm.s2_back += s2_back_n

    # Stack these terms onto the BHmie object along axis 0,
    # because they need to be referenced later
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

    NN   = bhm.an.shape[0]
    inds = np.arange(bhm.an.shape[0]) + 1.0
    EN   = np.repeat(np.repeat(inds.reshape(NN,1,1), NA, axis=1), NE, axis=2)
    assert np.sum(EN[:,0,0] - inds) == 0
    assert np.sum(EN[0,:,:] - 1.0) == 0  # make sure that the first layer is all ones

    a2_b2    = (2.0 * EN + 1.0) * (np.abs(bhm.an)**2 + np.abs(bhm.bn)**2)
    bhm.Qsca = (2.0 / x**2) * np.sum(a2_b2, 0)

    bhm.gsca  = 2.0 * np.sum(bhm.gsca_terms, 0) / bhm.Qsca

    bhm.Qext  = (4.0 / x**2) * bhm.s1_ext.real

    bhm.Qback = (np.abs(bhm.s1_back) / x)**2 / np.pi
    return

def _calc_diff(bhm):
    """
    Calculate differential scattering cross section for given angles
    Returns in units of cm^2 steridian^-1
    """
    NA, NE, NTH = bhm.NA, bhm.NE, bhm.NTH
    x_tile = np.repeat(bhm.X.reshape(NA, NE, 1), NTH, axis=2)
    s1_s2  = np.abs(bhm.S1)**2 + np.abs(bhm.S2)**2

    # Geometric cross section, in cm^2
    cgeo = np.pi * (bhm.a * c.micron2cm)**2  # NA x NE
    cgeo_tile = np.repeat(cgeo.reshape(NA, NE, 1), NTH, axis=2)

    # for debugging
    if bhm.NTH > 1:
        assert np.sum(x_tile[:,:,0] - x_tile[:,:,1]) == 0
        assert np.sum(cgeo_tile[:,:,0] - cgeo_tile[:,:,1]) == 0

    dQ = (0.5 * s1_s2) / (np.pi * x_tile**2)
    return dQ * cgeo_tile
