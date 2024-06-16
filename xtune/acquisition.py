# -*- coding:utf-8 -*-
# Author: xiansong@nj.iscas.ac.cn

import warnings
import numpy as np
from scipy.stats import norm

from .util import convert_configurations_to_array


class Acquisition(object):
    def __init__(self, model, time_model=None, best_perf=0, xi=0.001):
        self.model = model
        self.time_model = time_model
        self.xi = xi
        self.best_perf = best_perf

    def __call__(self, configurations):
        X = convert_configurations_to_array(configurations)
        X = np.asarray(X)
        mu, std = self._predict(self.model, X)
        acq_vals = self._compute(mu, std)
        inv_t = self._t_predict(X)
        acq_vals *= inv_t
        return acq_vals

    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def _predict(self, model, X):
        with warnings.catch_warnings():
            mu, std = model.predict(X)
        return mu, std

    def _t_predict(self, X):
        mu, std = self._predict(self.time_model, X)
        mu = (mu - mu.min()) / (mu.max() - mu.min())
        std = (std - std.min()) / (std.max() - std.min())
        inv_t = np.exp(-mu + 0.5 * std ** 2)
        return inv_t


    def _compute(self, mu, std):
        values = np.zeros_like(mu)
        mask = std > 0
        improve = self.best_perf - self.xi - mu[mask]
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        values[mask] = exploit + explore

        return -values
