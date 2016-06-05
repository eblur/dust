
from . import scatmodels
from . import sigma_scat

from scipy.interpolate import interp1d

class ExtinctionCurve(object):
    def __init__(self):
        self.wavel = None    # Angstroms
        self.tau_ext = None
        self.tau_sca = None
        self.tau_abs = None

    def interp_ext(self, new_wavel):
        if (self.wavel is not None) and (self.tau_ext is not None):
            print("Need to run tau_ext calculation")
            return
        else:
            intobj = interp1d(self.wavel, self.tau_ext)
            return intobj(new_wavel)

    def interp_sca(self, new_wavel):
        if (self.wavel is not None) and (self.tau_sca is not None):
            print("Need to run tau_sca calculation")
            return
        else:
            intobj = interp1d(self.wavel, self.tau_sca)
            return intobj(new_wavel)

    def interp_abs(self, new_wavel):
        if (self.wavel is not None) and (self.tau_abs is not None):
            print("Need to run tau_abs calculation")
            return
        else:
            intobj = interp1d(self.wavel, self.tau_abs)
            return intobj(new_wavel)
