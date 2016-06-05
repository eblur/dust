from . import cmindex as cm

__all__ = ['Composition']

GTYPES = ['Graphite', 'Silicate', 'Drude']

# grain densities in g cm^-3
RHO_GRA = 2.2
RHO_SIL = 3.8
RHO_DRU = 3.0

RHOS   = dict(zip(GTYPES, [RHO_GRA, RHO_SIL, RHO_DRU]))
CMS    = dict(zip(GTYPES, [cm.CmGraphite(), cm.CmSilicate(), cm.CmDrude()]))

class Composition(object):
    """
    Composition class for storing information about grain material
        |
        | **ATTRIBUTES**
        |
    """
    def __init__(self, composition):
        if composition not in GTYPES:
            raise ValueError("Grain composition not recognized")
            return
        self.cname = composition
        self.rho   = RHOS[composition]

        if composition == 'Graphite':
            self.cmindex = cm.CmGraphite()
        if composition == 'Silicate':
            self.cmindex = cm.CmSilicate()
        if composition == 'Drude':
            self.cmindex = cm.CmDrude(rho=self.rho)
