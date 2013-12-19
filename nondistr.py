#!/usr/bin python2
# -*- coding: utf-8 -*-

import math
import triangle
import numpy as np
from scipy import special
import sys
sys.path.append('/home/ilya/work/emcee')
import emcee


class LnLike(object):
    """
    Class that represents likelihood using the model parameters from model:

        R = SN_{+} / SN_{||} = p[0] * || p[1] * np.exp(1j * p[2]) +
                                         p[3] *  np.exp(1j * p[4]) +
                                         p[5] * np.exp(1j * p[6]) ||

    for data of amplitudes of cross-to-parallel hands ratios (using SN-ratios
    as a proxies) and upper limits on them. Also use SN_{||} for calculating
    errors on R and gaussian (or t-distribution) noise distribution.

    """

    def __init__(self, detections, ulimits):
        """
        detections - np.array (#detections, 2,) values of detected amplitudes of
            cross-to-parallel hands ratios and corresponding signal-to-noise
            ratios of the parallel hands (for noise std calculations),
        ulimits - np.array (#ulimits, 2,) values of upper limits on amplitudes
            of cross-to-parallel hands ratios and corresponding signal-to-noise
            ratios of the parallel hands.
        """

        self.detections = detections
        self.ulimits = ulimits
        self.sigma_det = np.sqrt(1 + detections[:, 0] ** 2) / detections[:, 1]
        self.sigma_uli = np.sqrt(1 + ulimits[:, 0] ** 2) / ulimits[:, 1]

    def __call__(self, p):
        #ln of likelihood
        lnlik = (-0.5 * np.log(2 * math.pi * self.sigma_det ** 2) -
               (self.detections[:, 0] -
                self.model(p)) ** 2 / (2. * self.sigma_det ** 2)).sum() +\
              (-np.log(2.) + np.log(1. + special.erf((self.ulimits[:, 0] -
               self.model(p)) / (math.sqrt(2.) * self.sigma_uli)))).sum()

        return lnlik

    def model(self, p):
        """
        Method that given parameter vector (walker) returns of cross-to-parallel
        hands ratio.

        Parameters:

            p - walker (7,).

        Output:

            value of cross-to-parallel hands ratio.
        """

        result = p[1] * np.exp(1j * p[2]) + p[3] * np.exp(1j * p[4]) +\
                 p[5] * np.exp(1j * p[6])

        return p[0] * np.sqrt((result * result.conjugate()).real)

    def model_vectorized(self, p):
        """
        Method that given ensemble of walkers returns cross-to-parallel hands
        ratios.

        Parameters:

            p - ensemble of walkers (#walkers, #dim,).

        Output:

            np.array of (#walkers,) values of cross-hand ratios.
        """

        p = p.T

        return self.model(p[:, np.newaxis])


class LnPrior(object):
    """
    Class that represents the prior density of the model:

        R = SN_{+} / SN_{||} = p[0] * || p[1] * np.exp(1j * p[2]) +
                                         p[3] *  np.exp(1j * p[4]) +
                                         p[5] * np.exp(1j * p[6]) ||
    """

    # TODO: implement choosing parameters of distributions
    def __init__(self, kind='skeptical'):
        """
        Inputs:

        Output:

        """

        self._allowed_kinds = ['skeptical', 'lowpol', 'highpol']
        if kind not in self._allowed_kinds:
            raise Exception('``kind`` must be in ' + str(self._allowed_kinds))
        self.kind = kind

        if kind is 'skeptical':

            self.r_mu = 0
            self.r_sigma = 0.1

            self.m_min = 0.0
            self.m_max = 0.05

            self.dgrt_min = 0.01
            self.dgrt_max = 0.15

            self.d_min = 0.00
            self.d_max = 0.2

        elif kind is 'lowpol':

            self.r_mu = 0
            self.r_sigma = 0.25

            self.m_min = 0
            self.m_max = 0.025

            self.dgrt_min = 0.01
            self.dgrt_max = 0.2

            self.d_min = 0.08
            self.d_max = 0.12

        elif kind is 'highpol':

            self.r_mu = 0
            self.r_sigma = 0.25

            self.m_min = 0.025
            self.m_max = 0.10

            self.dgrt_min = 0.01
            self.dgrt_max = 0.2

            self.d_min = 0.08
            self.d_max = 0.12

    def __call__(self, p):
        """
        Inputs:

        Output:

        """

        p = p.T

        result = vec_lnlognorm(p[0], self.r_mu, self.r_sigma) +\
                 vec_lngenbeta(p[1], 2, 3, self.m_min, self.m_max) +\
                 vec_lnunif(p[2], -math.pi, math.pi) +\
                 vec_lngenbeta(p[3], 3, 8, self.dgrt_min, self.dgrt_max) +\
                 vec_lnunif(p[4], -math.pi, math.pi) +\
                 vec_lnunif(p[5], self.d_min, self.d_max) +\
                 vec_lnunif(p[6], -math.pi, math.pi)

        return result


class LnPost(object):
    """
    Class that represents the posterior density of the model:

        R = SN_{+} / SN_{||} = p[0] * || p[1] * np.exp(1j * p[2]) +
                                         p[3] *  np.exp(1j * p[4]) +
                                         p[5] * np.exp(1j * p[6]) ||
    """

    def __init__(self, detections, ulimits, prior_kind=None):
        self._lnpr = LnPrior(kind=prior_kind)
        self._lnlike = LnLike(detections, ulimits)

    def lnpr(self, p):
        return self._lnpr(p)

    def lnlike(self, p):
        return self._lnlike.__call__(p)

    def __call__(self, p):
        return self.lnlike(p) + self.lnpr(p)


#scipy.stats.lognorm.logpdf(x, i don't know:))
def vec_lnlognorm(x, mu, sigma):
    """
    Vectorized (natural logarithm of) LogNormal distribution.

    Input:

        ``x`` - value or numpy.ndarray of values of the parameter.

    Output:

        numpy.ndarray with size of ``x``.
    """

    x_ = np.where(0 < x, x, 1)
    result1 = -np.log(np.sqrt(2. * math.pi) * x_ * sigma) - (np.log(x_) - mu)\
        ** 2 / (2. * sigma ** 2)
    result = np.where(0 < x, result1, float("-inf"))

    return result


def vec_lngenbeta(x, alpha, beta, c, d):
    """
    Vectorized (natural logarithm of) Beta distribution with support (c,d).
    A.k.a. generalized (4-parametric) Beta distribution.

    Input:

        ``x`` - value or numpy.ndarray of values of the parameter.

    Output:

        numpy.ndarray with size of ``x``.
    """

    x_ = np.where((c < x) & (x < d), x, 1)

    result1 = -math.log(special.beta(alpha, beta)) - (alpha + beta - 1.) *\
        math.log(d - c) + (alpha - 1.) * np.log(x_ - c) + (beta - 1.) *\
        np.log(d - x_)
    result = np.where((c < x) & (x < d), result1, float("-inf"))

    return result


#scipy.stats.uniform.logpdf(x, a, b - a)
def vec_lnunif(x, a, b):
    """
    Vectorized (natural logarithm of) uniform distribution on [a, b].

    Input:

        ``x`` - value or numpy.ndarray of values of the parameter.

    Output:

        numpy.ndarray with size of ``x``.
    """

    result1 = -np.log(b - a)
    result = np.where((a <= x) & (x <= b), result1, float('-inf'))

    return result


if __name__ == '__main__()':

    # Create big dataset
    lnpri = LnPrior(kind='lowpol')
    #lnpri = LnPrior(kind='highpol')

    # Initialize workers
    ndim = 7
    nwalkers = 500

    r1 = np.random.uniform(low=0.05, high=0.15, size=(nwalkers, 1))
    # Initial value for low polarization case
    r2 = np.random.uniform(low=0.0, high=0.025, size=(nwalkers, 1))
    # For high polarization
    #r2 = np.random.uniform(low=0.025, high=0.10, size=(nwalkers, 1))
    r3 = np.random.uniform(low=-3, high=3, size=(nwalkers, 1))
    r4 = np.random.uniform(low=0.01, high=0.2, size=(nwalkers, 1))
    r5 = np.random.uniform(low=-3, high=3, size=(nwalkers, 1))
    r6 = np.random.uniform(low=0.08, high=0.15, size=(nwalkers, 1))
    r7 = np.random.uniform(low=-3, high=3, size=(nwalkers, 1))

    p0 = np.hstack((r1, r2, r3, r4, r5, r6, r7))

    # Will sample from multidim. prior. It reflects our hypothesis.
    lnpost = lnpri

    # Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, bcast=True)
    # Burn-in
    pos, prob, state = sampler.run_mcmc(p0, 10000)
    sampler.reset()

    # Run sampler
    sampler.run_mcmc(pos, 1000)

    # Generated 2500 independent samples from prior
    p_gen = sampler.flatchain[::100]

    # Check prior
    figure = triangle.corner(p_gen)

    ## C band D_L. Actually, we need them just to initialize LnLike instance
    detections = np.array([[0.143, 50.], [0.231, 50], [0.077, 50], [0.09, 50],
                           [0.152, 50], [0.115, 50], [0.1432, 50], [0.1696, 50],
                           [0.1528, 50], [0.126, 50], [0.1126, 50], [0.138, 50],
                           [0.194, 50], [0.109, 50], [0.101, 50]])
    ulimits = np.array([[0.175, 33], [0.17, 34], [0.17, 34], [0.088, 65],
                        [0.187, 30], [0.1643, 35], [0.0876, 65], [0.123, 46],
                        [0.057, 100], [0.155, 37]])



    # Trying to estimate from real data
    # Using affine-invariant MCMC
    nwalkers = 500
    ndim = 7
    lnpost = LnPost(detections, ulimits, prior_kind='skeptical')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, bcast=True)
    pos, prob, state = sampler.run_mcmc(p0, 10000)
    sampler.reset()

    sampler.run_mcmc(pos, 5000)







    lnlik = LnLike(detections, ulimits)

    # First, we need distribution over the model parameters (hypothesis).
    # We can formulate hypothesis indirectly  by generating idealized data for
    # bayesian analysis and use the resulting posterior as hypothesis
    # 1) Generate idealized data
    data_gen = lnlik.model_vectorized(p_gen)[0]
    # Problem: Datasets don't differ!!! Model is not correct?
    # 2) Conduct bayesian analysis to get posterior - our hypothesis
    # Initialize workers
    ndim = 7
    nwalkers = 500
    p0 = np.random.uniform(low=0.02, high=0.1, size=(nwalkers, ndim))

    lnpri = LnPrior()


    #detections_gen =

    # Or we can use hypothesize on model parameters directly and use their
    # distribution as hypothesis
