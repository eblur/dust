
"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# eblur/dust specific imports

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import distlib
    from . import extinction
    from . import halos

# ----------------------------------------------------------------------------
