
a = np.linspace(0.005, 0.3, 50)
E = np.logspace(-1, 1, 20)

NA, NE = len(a), len(E)
amat   = np.tile(a, (NA, 1))
Emat   = np.tile(E, (NE, 1)).T

