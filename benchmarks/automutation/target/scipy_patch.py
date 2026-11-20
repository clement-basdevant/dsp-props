"""pytest plugin (``-p scipy_patch``): SciPy's own tests call the copies in ``scipy_subset``.

The upstream test module imports the functions from ``scipy.signal`` when it is collected, so the
patch is applied when this plugin is loaded, before collection.
"""

import scipy.signal
import scipy.signal._signaltools as _st
import scipy_subset
from scipy._lib._array_api import xp_capabilities_table

FUNCTIONS = ["resample", "resample_poly", "decimate", "hilbert"]

for _name in FUNCTIONS:
    # the array-API test markers look the function up in a table keyed by the function object
    xp_capabilities_table[getattr(scipy_subset, _name)] = xp_capabilities_table[getattr(scipy.signal, _name)]
    setattr(scipy.signal, _name, getattr(scipy_subset, _name))
    setattr(_st, _name, getattr(scipy_subset, _name))
