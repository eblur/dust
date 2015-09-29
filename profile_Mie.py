## profile_Mie.py -- A script for profiling the Mie scattering function
##
## 2015.09.29 -- lia@space.mit.edu
##----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import dust
import sigma_scat as ss


AMIN, AMAX = 0.005, 0.25 # microns
NA  = 50  # number of points to use to sample distribution
RHO = 3.8 # grain density (g cm^-3)
P   = 3.5 # power law slope
mrn = dust.Dustdist( rad=np.linspace(AMIN,AMAX,NA), rho=RHO, p=P )

ENERGY = np.logspace(-1,1,50)
MDUST  = 1.e22 * dust.c.mp() * 0.009  # magic numbers (dust mass per 10^22 H)


kappascat = ss.Kappascat(E=ENERGY, dist=dust.Dustspectrum(rad=mrn), 
                         scatm=ss.makeScatmodel('Mie','Silicate'))

kappascat()


