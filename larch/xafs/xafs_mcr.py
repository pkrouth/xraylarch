#!/usr/bin/env python
# XAS spectral decovolution
#

import numpy as np
from scipy.signal import deconvolve
from larch import parse_group_args, Make_CallArgs

from larch.math import (gaussian, lorentzian, interp,
                        index_of, index_nearest, remove_dups,
                        savitzky_golay)

from .xafsutils import set_xafsGroup

@Make_CallArgs(["energy","norm"])
def xas_mcr(energy, norm=None, group=None, form='lorentzian',
                   esigma=1.0, eshift=0.0, smooth=True,
                   sgwindow=None, sgorder=3, _larch=None):
    """XAS multivariate curve resolution
    ....
    de-convolve a normalized mu(E) spectra with a peak shape, enhancing the
    intensity and separation of peaks of a XANES spectrum.

    The results can be unstable, and noisy, and should be used
    with caution!

    Arguments
    ----------
    energy:   array of x-ray energies (in eV) or XAFS data group
    norm:     array of normalized mu(E)
    group:    output group
    form:     functional form of deconvolution function. One of
              'gaussian' or 'lorentzian' [default]
    esigma    energy sigma to pass to gaussian() or lorentzian()
              [in eV, default=1.0]
    eshift    energy shift to apply to result. [in eV, default=0]
    smooth    whether to smooth result with savitzky_golay method [True]
    sgwindow  window size for savitzky_golay [found from data step and esigma]
    sgorder   order for savitzky_golay [3]

    Returns
    """
    return 
