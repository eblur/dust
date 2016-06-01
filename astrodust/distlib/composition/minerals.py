
## minerals.py -- Some tables for ISM abundances and depletion factors
## that are useful for calculating dust mass and dust-to-gas ratios
##
## 2016.01.22 - lia@space.mit.edu
##----------------------------------------------------------------

import numpy as np


amu   = {'H':1.008,'He':4.0026,'C':12.011,'N':14.007,'O':15.999,'Ne':20.1797, \
         'Na':22.989,'Mg':24.305,'Al':26.981,'Si':28.085,'P':30.973,'S':32.06, \
         'Cl':35.45,'Ar':39.948,'Ca':40.078,'Ti':47.867,'Cr':51.9961,'Mn':54.938, \
         'Fe':55.845,'Co':58.933,'Ni':58.6934}
amu_g = 1.661e-24 # g
mp    = 1.673e-24 # g (proton mass)

wilms = {'H':12.0, 'He':10.99, 'C':8.38, 'N':7.88, 'O':8.69, 'Ne':7.94, \
               'Na':6.16, 'Mg':7.40, 'Al':6.33, 'Si':7.27, 'P':5.42, 'S':7.09, \
               'Cl':5.12, 'Ar':6.41, 'Ca':6.20, 'Ti':4.81, 'Cr':5.51, 'Mn':5.34, \
               'Fe':7.43, 'Co':4.92, 'Ni':6.05} # 12 + log A_z

# Fraction of elements still in gas form
wilms_1mbeta = {'H':1.0, 'He':1.0, 'C':0.5, 'N':1.0, 'O':0.6, 'Ne':1.0, 'Na':0.25, \
             'Mg':0.2, 'Al':0.02, 'Si':0.1, 'P':0.6, 'S':0.6, 'Cl':0.5, 'Ar':1.0, \
             'Ca':0.003, 'Ti':0.002, 'Cr':0.03, 'Mn':0.07, 'Fe':0.3, 'Co':0.05, \
             'Ni':0.04}

class Mineral(object):
    """
    Mineral object
    -------------------
    Use a dictionary to define the composition.
    e.g. Olivines of pure MgFe^{2+}SiO_4 composition would be
    olivine_halfMg = Mineral( {'Mg':1.0, 'Fe':1.0, 'Si':1.0, 'O':4.0} )
    -------------------
    self.composition : dictionary containing elements and their weights
    @property
    self._weight_amu : amu weight of unit crystal
    self.weight_g    : g weight of unit crystal
    """
    def __init__(self, comp):
        self.composition = comp

    @property
    def weight_amu(self):
        result = 0.0
        for atom in self.composition.keys():
            result += self.composition[atom] * amu[atom]
        return result

    @property
    def weight_g(self):
        return self.weight_amu * amu_g

def calc_mass_conversion( elem, mineral ):
    """
    calc_mass_conversion( elem, mineral )
    Returns the number of atoms per gram of a particular mineral object
    Useful for converting mass column to a number density column for an element
    """
    assert type(mineral) == Mineral
    assert type(elem) == str
    return mineral.composition[elem] / mineral.weight_g  # g^{-1}


def calc_element_column( NH, fmineral, atom, mineral, d2g=0.009 ):
    """
    Calculate the column density of an element for a particular NH value,
    assuming a dust-to-gas ratio (d2g) and
    the fraction of dust in that particular mineral species (fmineral)
    --------------------------------------------------------------------
    calc_element_column( NH, fmineral, atom, mineral, d2g=0.009 )
    """
    dust_mass = NH * mp * d2g * fmineral # g cm^{-2}
    print('Dust mass = %.3e g cm^-2' % (dust_mass))
    return calc_mass_conversion(atom, mineral) * dust_mass # cm^{-2}


def get_ISM_abund(elem, abund_table=wilms):
    """
    get_ISM_abund( elem, abund_table )
    ----
    Given an abundance table, calculate the number per H atom of a
    given element in any ISM form
    """
    assert type(elem) == str
    assert type(abund_table) == dict
    return np.power(10.0, abund_table[elem] - 12.0)  # number per H atom

def get_dust_abund(elem, abund_table=wilms, gas_ratio=wilms_1mbeta):
    """
    get_dust_abund( elem, abund_table, gas_ratio)
    ----
    Given an abundance table (dict) and a table of gas ratios (dict),
    calculate the number per H atom of a given ISM element in *solid* form
    """
    assert type(elem) == str
    assert type(abund_table) == dict
    assert type(gas_ratio) == dict
    return get_ISM_abund(elem, abund_table) * (1.0 - gas_ratio[elem]) # number per H atom
