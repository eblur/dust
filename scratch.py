
import numpy as np
import cmindex as cmi

a = np.linspace(0.005, 0.3, 50)
E = np.logspace(-1, 1, 20)

NA, NE = np.size(a), np.size(E)
amat   = np.tile(a, (NE, 1))
Emat   = np.tile(E, (NA, 1)).T

X = 2.0*np.pi*amat / Emat

# See that cmindex works fine interpolating over Emat
cm  = cmi.CmGraphite()
rptest = cm.rp(Emat)

## Start executing soem of the main code to see how it works

refrel  = cm.rp(Emat) + 1j*cm.ip(Emat)

x      = X
y      = X * refrel
ymod   = np.abs(y)
nx     = np.size( X )


# *** Series expansion terminated after NSTOP terms
# Logarithmic derivatives calculated from NMX on down

xstop  = X + 4.0 * np.power( X, 0.3333 ) + 2.0
test   = np.append( xstop, ymod )
nmx    = np.max( test ) + 15
nmx    = np.int32(nmx)

d = np.zeros( shape=(NA,NE,nmx+1), dtype='complex' )
dold = np.zeros( nmx+1, dtype='complex' )
