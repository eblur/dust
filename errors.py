
import numpy as np

def prop_add( xerr=0.0, yerr=0.0 ):
	return np.sqrt( xerr**2 + yerr**2 )

def prop_div( x, y, xerr=0.0, yerr=0.0 ):
	F = x / y
	return np.sqrt( xerr**2 + F**2 * yerr**2 ) / y

def prop_mult( x, y, xerr=0.0, yerr=0.0 ):
	F = x * y
	return np.sqrt( (xerr/x)**2 + (yerr/y)**2 ) * F
	
"""
## Quick test
import matplotlib.pyplot as plt

i = np.arange(10.0) + 1
x = 2.0 * i
y = 4.0 * i
xerr = np.zeros( 10.0 ) + 5.0
yerr = np.zeros( 10.0 ) + 2.5

test_add  = prop_add( xerr, yerr )
test_div  = prop_div( x, y, xerr, yerr )
test_mult = prop_mult( x, y, xerr, yerr )

fig = plt.figure()
plt.errorbar( i, x, yerr=xerr, ls='', color='r', lw=3, alpha=0.3 )
plt.errorbar( i, y, yerr=yerr, ls='', color='b', lw=3, alpha=0.3 )
plt.errorbar( i, x + y, yerr=test_add, ls='', color='0.3', lw=3, alpha=0.3 )
plt.errorbar( i, x / y, yerr=test_div, ls='', color='0.5', lw=3, alpha=0.3 )
plt.errorbar( i, x * y, yerr=test_mult, ls='', color='0.7', lw=3, alpha=0.3 )
plt.ylim(-1,1)
fig.show()

print('Add, no yerr:', prop_add( xerr ))
print('Add, no xerr:', prop_add( yerr=yerr ))
print('Div, no yerr:', prop_div( x, y, xerr ))
print('Should be:', xerr / y)
print('Div, no xerr:', prop_div( x, y, yerr=yerr ))
print('Mult, no yerr:', prop_mult( x, y, xerr ))
print('Should be:', y * xerr)
print('Mult, no xerr:', prop_mult( x, y, yerr=yerr ))
print('Should be:', x * yerr)
"""
