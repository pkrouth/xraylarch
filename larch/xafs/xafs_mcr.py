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
# Create MCR Object
import sys
import logging

import seaborn as sns

#sns.set_theme(context="notebook", style="ticks", palette="Set1")


import matplotlib.pyplot as plt

#plt.rcParams["savefig.dpi"] = 100
import numpy as np
import pandas as pd


# Importing pymcr pieces
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import (
    ConstraintNonneg,
    ConstraintNorm,
    ConstraintZeroCumSumEndPoints,
)

from sklearn.linear_model import Ridge

# Object Oriented Imports
from dataclasses import dataclass
from typing import List
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA

from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler, StandardScaler






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



@dataclass
class MCR_ALS:
    D: np.array
    fit_params: "typing.Any"
    exp_params: "typing.Any"
    initial_spectra: np.array = None
    initial_conc_map: np.array = None

    def initialize(self):
        if self.fit_params.NORMALIZE:
            self.D_scaler = StandardScaler().fit(X=self.D)
            # print(f"Min max normalization of Dataset \'D\' with {self.D_scaler.data_max_}")
            self.D_scaled = self.D_scaler.transform(self.D)

        if self.fit_params.SP_GUESS and self.fit_params.SP_GUESS_TYPE == "RAND":
            np.random.seed(self.fit_params.RANDOM_STATE)
            self.initial_spectra = np.random.rand(self.exp_params.p, self.exp_params.n)
            print(
                f"Initial Guess of Spectra with Random numbers. Shape of guess{self.initial_spectra.shape}"
            )

        elif self.fit_params.SP_GUESS and self.fit_params.SP_GUESS_TYPE == "SVD":
            U, s, Vh = svds(self.D, k=self.exp_params.p + 1)
            Vh = Vh[np.flip(np.argsort(s))[:-1], :]
            self.initial_spectra = np.abs(Vh) / Vh.max() * self.D.max()
            print(
                f"Initial Guess of Spectra with SVD. Shape of guess{self.initial_spectra.shape}"
            )

        elif self.fit_params.SP_GUESS and self.fit_params.SP_GUESS_TYPE == "PCA":
            self._pca = PCA(self.exp_params.p)
            if self.fit_params.NORMALIZE:
                self._pca.fit(self.D_scaled)
                self.initial_spectra = self._pca.components_

            else:
                self._pca.fit(self.D)
                self.D_PrePCA = self.D  # saved original D matrix
                self.initial_spectra = self._pca.components_
                # self.initial_spectra = self._pca.transform(self.D)#.transpose()
            # init_scaler = MinMaxScaler().fit(X=self.initial_spectra.T)
            # self.initial_spectra = init_scaler.transform(self.initial_spectra.T).transpose()

        elif self.fit_params.CONC_GUESS:
            print("Guessing Concentration: Random")
            self.initial_conc_map = np.random.rand(self.exp_params.m, self.exp_params.p)
            print("Adding constraint of Sum to one in random guess")
            self.initial_conc_map = (
                self.initial_conc_map.T / (self.initial_conc_map.sum(axis=1)).T
            ).T

    def process_MCR(self, constraints, verbose=True, *args, **kwargs):
        self.mcrar = McrAR(
            max_iter=constraints.MAX_ITERATIONS,
            tol_increase=2,
            c_regr=constraints.C_regressor,
            st_regr=constraints.S_regressor,
            c_constraints=constraints.C_constraints,
            st_constraints=constraints.S_constraints,
            *args,
            **kwargs,
        )

        if self.fit_params.SP_GUESS:
            self.mcrar.fit(self.D, ST=self.initial_spectra, verbose=verbose)
        if self.fit_params.CONC_GUESS:
            self.mcrar.fit(self.D, C=self.initial_conc_map, verbose=verbose)
        print("\nFinal MSE: {:.7e}".format(self.mcrar.err[-1]))

        return self.mcrar

    def visualize_initial_guesses(self):
        if self.fit_params.SP_GUESS:
            if self.fit_params.NORMALIZE and self.D_scaler is not None:
                # guess_spectra = self.D_scaler.inverse_transform(self.initial_spectra)
                guess_spectra = self.initial_spectra
            else:
                guess_spectra = self.initial_spectra
            _ = plt.figure()
            for i in range(self.exp_params.p):
                _ = sns.lineplot(
                    x=list(range(self.exp_params.n)), y=guess_spectra[i, :]
                )
            plt.title(f"Spectra obtained by {self.fit_params.SP_GUESS_TYPE} ")
            _ = plt.show()
        else:
            self._visualize_concentration_maps(conc=self.initial_conc_map)

    def visualize_mcr_output(self, c_x=None, s_x=None):
        self._visualize_concentration_maps(conc=self.mcrar.C_opt_, x_axis=c_x)
        self._visualize_ST_opt_(x_axis=s_x)

    def _visualize_concentration_maps(self, conc, x_axis=None):
        if x_axis is None:
            x_axis = list(range(1, self.exp_params.m + 1, 1))
        _ = plt.figure()
        for i in range(self.exp_params.p):
            _ = sns.lineplot(x=x_axis, y=conc[:, i])
        _ = plt.title("Concentration map obtained by pyMCR")
        _ = plt.show()

    def _visualize_ST_opt_(self, x_axis=None):
        if x_axis is None:
            x_axis = list(range(self.exp_params.n))
        _ = plt.figure()
        for i in range(self.exp_params.p):
            ax = sns.lineplot(
                x=x_axis,
                y=self.mcrar.ST_opt_[i, :],
            )
        _ = ax.set_title("Spectra obtained by pyMCR")
        _ = plt.show()

    def visualize_data(self):
        raise Exception("Not Implemented Yet")

    def export_output(self, filename, path="./"):
        conc = pd.DataFrame(
            self.mcrar.C_opt_,
            columns=[f"Sp{i}" for i in range(1, self.exp_params.p + 1)],
        )
        conc["residual"] = conc.sum(axis=1) - 1
        conc.to_csv(path + filename + "_conc.csv", index=False)
        spectra = pd.DataFrame(
            self.mcrar.ST_opt_.T,
            columns=[f"Sp{i}" for i in range(1, self.exp_params.p + 1)],
        )
        spectra.to_csv(path + filename + "_spectra.csv", index=False)


if __name__ == "__main__":
    pass
    # assert SP_GUESS is not CONC_GUESS


@dataclass
class McrConstraints:
    C_constraints: "typing.List" = field(
        default_factory=lambda: [
            ConstraintNorm(),
            ConstraintNonneg(),
        ]
    )  #
    S_constraints: "typing.List" = field(default_factory=lambda: [])  #
    C_regressor: str = "NNLS"
    S_regressor: str = "NNLS"
    MAX_ITERATIONS: int = 2000


@dataclass
class FittingParams:
    NORMALIZE: bool = True
    SP_GUESS: bool = False
    CONC_GUESS: bool = field(init=False, repr=True)
    SP_GUESS_TYPE: str = "SVD"  # RAND, SVD, PCA
    RANDOM_STATE: int = 1200

    def __post_init__(self):
        self.CONC_GUESS = not self.SP_GUESS


@dataclass
class ExperimentParams:
    p: int  # number of species
    m: int  # number of measurements
    n: int  # number of bins


def run_mcr(X, filename, fit_params, experiment_params, fit_constraints, verbose=False):
    mcr = MCR_ALS(X.T, fit_params, experiment_params)
    # Initialize according to FLAGS mentioned above
    mcr.initialize()
    mcr.visualize_initial_guesses()
    mcr.process_MCR(constraints=fit_constraints, verbose=verbose)
    mcr.visualize_mcr_output()
    mcr.export_output(filename=filename)
    return mcr

