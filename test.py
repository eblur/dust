import numpy as np
import matplotlib.pyplot as plt

print('importing...')
import sigma_scat as ss
import dust
import constants as c

print('setting input...')
# column density
NH   = 1.e21
# dust-to-gas ratio
d2g = 0.009
# use proton mass to get dust column mass
dust_mass = NH * c.mp() * d2g 
# energy range to evaluate over (in keV)
ERANGE    = np.power( 10.0, np.arange(-0.6,1.0,0.005) )
MD = dust_mass * 10.0 # for N_H = 1.e22

# define densities
# g cm^-3; see Draine book
RHO_SIL, RHO_GRA, RHO_AVG = 3.8, 2.2, 3.0 

print('defining dust distributions...')

# dust radius range to compute (in um)
# up to 0.25 um
MRN_RANGE = np.arange(0.005,0.25001,0.05)

# larger range going up to 1.5 um
BIG_RANGE = np.arange(0.005, 1.5, 0.05)
MRN_RANGE = BIG_RANGE
# define dust distribution: A powerlaw with index 3.5
MRN_sil   = dust.Dustdist( rad=MRN_RANGE, p=3.5, rho=RHO_SIL )
MRN_gra   = dust.Dustdist( rad=MRN_RANGE, p=3.5, rho=RHO_GRA )
MRN_avg   = dust.Dustdist( rad=MRN_RANGE, p=3.5, rho=RHO_AVG )

print('Defining Kappaext and Dustspectrum...')

print('    Mie scattering with Drude approximation')
RGD_mrn = ss.Kappaext( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_avg, md=dust_mass), scatm=ss.makeScatmodel('Mie','Drude') )

print('    Mie scattering for the small grain MRN distribution')
Mie_mrn_sil = ss.Kappaext( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_sil, md=dust_mass), scatm=ss.makeScatmodel('Mie','Silicate') )
Mie_mrn_gra = ss.Kappaext( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_gra, md=dust_mass), scatm=ss.makeScatmodel('Mie','Graphite') )

print('plotting...')

plt.plot( RGD_mrn.E, RGD_mrn.kappa * MD, '0.4', lw=2, label='Mie-Drude' )
plt.plot( Mie_mrn_sil.E, Mie_mrn_sil.kappa * MD, 'g', lw=2, label='Mie-Silicate' )
plt.plot( Mie_mrn_gra.E, Mie_mrn_gra.kappa * MD, 'b', lw=2, label='Mie-Graphite' )

plt.legend( loc='upper right', fontsize=12 )

plt.loglog()
plt.xlim(0.3,10)
plt.ylim(1.e-2,10.0)
plt.xlabel( "Energy [keV]", size=15 )
plt.ylabel( r"Scattering Opacity [$\tau$ per $N_{\rm H}/10^{22}$]", size=15 )

plt.savefig('figure5a.pdf', format='pdf')
plt.close()
np.savetxt('figure5a.txt', np.transpose([ERANGE, RGD_mrn.kappa, Mie_mrn_sil.kappa, Mie_mrn_gra.kappa]))

################################################################################

print('Defining Kappascat and Dustspectrum...')

print('    Rayleigh-Gans plus Drude approximation')
RGD_mrn = ss.Kappascat( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_avg, md=dust_mass), scatm=ss.makeScatmodel('RG','Drude') )

print('    Mie scattering for the small grain MRN distribution')
Mie_mrn_sil = ss.Kappascat( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_sil, md=dust_mass), scatm=ss.makeScatmodel('Mie','Silicate') )
Mie_mrn_gra = ss.Kappascat( E=ERANGE, dist=dust.Dustspectrum(rad=MRN_gra, md=dust_mass), scatm=ss.makeScatmodel('Mie','Graphite') )

print('plotting...')


plt.plot( RGD_mrn.E, RGD_mrn.kappa * MD, '0.4', lw=2, label='RG-Drude' )
plt.plot( Mie_mrn_sil.E, Mie_mrn_sil.kappa * MD, 'g', lw=2, label='Mie-Silicate' )
plt.plot( Mie_mrn_gra.E, Mie_mrn_gra.kappa * MD, 'b', lw=2, label='Mie-Graphite' )

plt.legend( loc='upper right', fontsize=12 )


plt.loglog()
plt.xlim(0.3,10)
plt.ylim(1.e-2,10.0)
plt.xlabel( "Energy [keV]", size=15 )
plt.ylabel( r"Scattering Opacity [$\tau$ per $N_{\rm H}/10^{22}$]", size=15 )

plt.text( 0.5, 0.1, '$0.25\ \mu m$\ncut-off', size=12)
plt.text( 1.2, 3.0, '$1.5\ \mu m$\ncut-off', size=12)
plt.savefig('figure5b.pdf', format='pdf')
plt.close()
np.savetxt('figure5b.txt', np.transpose([ERANGE, RGD_mrn.kappa, Mie_mrn_sil.kappa, Mie_mrn_gra.kappa]))

