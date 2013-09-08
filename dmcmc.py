#!/usr/bin python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/ilya/work/emcee')
import math
import emcee
import numpy as np
from scipy import special
from knuth_hist import histogram
from matplotlib.pyplot import bar


class LnPost(object):
    """
    Class that represents posterior density of given amplitude of D-.
    Using ``detections`` and ``ulimits`` parameters find probability of them
    being generated from cross-to-parallel hands distribution generated
    by distributions of other parameters and given amplitude of the
    D-term ``d``.

    """

    def __init__(self, detections, ulimits, distributions, lnpr=None, args=None):
        """
        Parameters:

            detections - values of cross-to-parallel hands ratios for the
                         detected cross-hands,

            ulimits - upper limits on cross-to-parallel hands ratios

            distributions - np.array (5, N) or list of 5 arrays (N,).
                            Distributions of (|r|, |M|, fi_M, |D_2|, fi_2, fi_1),

            lnpr - callable prior on amplitude of the D-term.

            args - list of the additional arguments for lnpr callable.
        """

        self.detections = detections
        self.ulimits = ulimits
        self.distributions = distributions
        self._lnpr = lnpr
        self._args = args

    def __call__(self, d):
        """
        Returns posterior probability of d.
        """

        ratio_distribution = self.model(d)
        print "Done modeling ratio distibution!"
        lnlks_detections = self.lnprob(self.detections, ratio_distribution)
        print "Ln of prob. for detections is : " + str(lnlks_detections)
        lnlks_ulimits = self.lnprob(self.ulimits, ratio_distribution, kind='u')
        print "Ln of prob. for ulimits is : " + str(lnlks_ulimits)
        try:
            lnlks = lnlks_detections + lnlks_ulimits
            lnlk = lnlks.sum()
            result = lnlk + self.lnpr(d)
        except TypeError:
            result = None

        return result

    def model(self, d):
        """
        Method that given amplitude of the D-term returns distribution of
        cross-to-parallel hands ratios.

        Parameters:

            d - amplitude of the D-term,

        Output:

            np.array of N values of cross-hand ratios.
        """

        data = self.distributions

        result = data[1] * np.exp(1j * data[2]) + data[3] * np.exp(1j * data[4])\
                 + d * np.exp(1j * data[5])

        return data[0] * np.sqrt((result * result.conjugate()).real)

    def lnprob(self, xs, distribution, kind=None):
        """
        Method that given some values ``xs`` and sample from some distribution
        (container of values) returns natural logarithm of likelihood of values
        ``xs`` being generated from given distribution.

        Parameters:

            xs - values of interest,

            distribution - [container] - sample from distribution that is checked,

            kind [None, 'u', 'l'] - type of values ``xs`` (detections, upper or
            lower limits).

        Output:

        Natural logarithm of likelihood of ``xs`` being generated from
        sample's distribution.
        """


        hist_d, edges_d = histogram(distribution, normed=True)

        lower_d = np.resize(edges_d, len(edges_d) - 1)
        knuth_width = np.diff(lower_d)[0]
        probs = np.zeros(len(xs))

        if kind is None:
            for i in range(len(probs)):
                probs[i] = knuth_width *\
                           hist_d[((lower_d[:, np.newaxis] - xs).T > 0)[i]][0]
        elif kind is 'u':
            for i in range(len(probs)):
                probs[i] = knuth_width *\
                            hist_d[np.where(((lower_d[:, np.newaxis] -\
                                              xs).T)[i] < 0)].sum()
        elif kind is 'l':
            raise NotImplementedError('Please, implement lower limits!')
        else:
            raise Exception('``kind`` parameter must be ``None``, ``u``\
                            or ``l``.')

        result = np.log(probs).sum()

        return result

    def lnpr(self, d):
        """
        Prior on D-term amplitude.
        """

        return self._lnpr(d, *self._args)

    
class LnLike(object):
    """
    Class used for MLE estimates.
    """
    
    def __init__(self, detections, ulimits, distributions):
        """
        Parameters:

            detections - values of cross-to-parallel hands ratios for the
                         detected cross-hands,

            ulimits - upper limits on cross-to-parallel hands ratios

            distributions - np.array (5, N) or list of 5 arrays (N,).
                            Distributions of (|r|, |M|, fi_M, |D_2|, fi_2, fi_1).
        """

        self.detections = detections
        self.ulimits = ulimits
        self.distributions = distributions

    def __call__(self, d):
        """
        Returns lnlikelihood of detections and ulimits being generated from
        distributions generated by model with given models distributions and given
        d.
        """

        ratio_distribution = self.model(d)
        print "Done modeling ratio distibution!"
        lnlks_detections = self.lnprob(self.detections, ratio_distribution)
        print "Ln of prob. for detections is : " + str(lnlks_detections)
        lnlks_ulimits = self.lnprob(self.ulimits, ratio_distribution, kind='u')
        print "Ln of prob. for ulimits is : " + str(lnlks_ulimits)
        
        lnlks = lnlks_detections + lnlks_ulimits
        lnlk = lnlks.sum()

        return lnlk

    def model(self, d):
        """
        Method that given amplitude of the D-term returns distribution of
        cross-to-parallel hands ratios.

        Parameters:

            d - amplitude of the D-term,

        Output:

            np.array of N values of cross-hand ratios.
        """

        data = self.distributions

        result = data[1] * np.exp(1j * data[2]) + data[3] * np.exp(1j * data[4])\
                 + d * np.exp(1j * data[5])

        return data[0] * np.sqrt((result * result.conjugate()).real)

    def lnprob(self, xs, distribution, kind=None):
        """
        Method that given some values ``xs`` and sample from some distribution
        (container of values) returns natural logarithm of likelihood of values
        ``xs`` being generated from given distribution.

        Parameters:

            xs - values of interest,

            distribution - [container] - sample from distribution that is checked,

            kind [None, 'u', 'l'] - type of values ``xs`` (detections, upper or
            lower limits).

        Output:

        Natural logarithm of likelihood of ``xs`` being generated from
        sample's distribution.
        """

        hist_d, edges_d = histogram(distribution, normed=True)

        lower_d = np.resize(edges_d, len(edges_d) - 1)
        knuth_width = np.diff(lower_d)[0]
        probs = np.zeros(len(xs))

        if kind is None:
            for i in range(len(probs)):
                probs[i] = knuth_width *\
                           hist_d[((lower_d[:, np.newaxis] - xs).T > 0)[i]][0]
        elif kind is 'u':
            for i in range(len(probs)):
                probs[i] = knuth_width *\
                            hist_d[np.where(((lower_d[:, np.newaxis] -\
                                              xs).T)[i] < 0)].sum()
        elif kind is 'l':
            raise NotImplementedError('Please, implement lower limits!')
        else:
            raise Exception('``kind`` parameter must be ``None``, ``u``\
                            or ``l``.')

        result = np.log(probs).sum()

        return result
    
    
class GaussProposal(object):
    
    def __init__(self, cov, nwalkers=1):
        
        self.nwalkers = int(nwalkers)
        
        self.cov = cov
        
        try:
            self.ndim = shape(cov)[0]
        except IndexError:
            self.ndim = 1

    def __call__(self, p):
        """
        p - list of positions of ``nwalker`` walkers.
        """
        
        nwalkers, ndim = p.shape
        
        assert(shape(p)[0] == self.nwalkers)
    
        if self.ndim == 1:
            if self.nwalkers == 1:
                result = [np.random.normal(loc=p, scale=self.cov)]
            else:
                result = [np.random.normal(loc=p_i, scale=self.cov) for p_i in p]
        else:
            if self.nwalkers == 1:
                result = [np.random.multivariate_normal(mean=p, cov=self.cov)]
            else:
                result = [np.random.multivariate_normal(mean=p_i, cov=self.cov) for
                          p_i in p]
                
        return result
    
    
def lnunif(x, a, b):
    """
    (Natural logarithm of) uniform distribution on [a, b].
    """

    result = - math.log(b - a)
    if not (a <= x) & (x <= b):
        result = float('-inf')

    return result


def vec_lnunif(x, a, b):
    """
    Vectorized (natural logarithm of) uniform distribution on [a, b].
    """

    result1 = -np.log(b - a)
    result = np.where((a <= x) & (x <= b), result1, float('-inf'))

    return result


def vec_lnlognorm(x, mu, sigma):
    """
    Vectorized (natural logarithm of) LogNormal distribution.
    """

    x_ = np.where(0 < x, x, 1)
    result1 = -np.log(np.sqrt(2. * math.pi) * x_ * sigma) - (np.log(x_) - mu)\
        ** 2 / (2. * sigma ** 2)
    result = np.where(0 < x, result1, float("-inf"))

    return result


def vec_lngenbeta(x, alpha, beta, c, d):
    """
    Vectorized (natural logarithm of) Beta distribution with support (c,d). A.k.a.
    generalized Beta distribution.
    """

    x_ = np.where((c < x) & (x < d), x, 1)

    result1 = -math.log(special.beta(alpha, beta)) - (alpha + beta - 1.) *\
        math.log(d - c) + (alpha - 1.) * np.log(x_ - c) + (beta - 1.) *\
        np.log(d - x_)
    result = np.where((c < x) & (x < d), result1, float("-inf"))

    return result


def logl(x):
    
    return lnunif(x, 0., 1.)


if __name__ == '__main__()':

    # C band D_L
    detections = [0.143, 0.231, 0.077, 0.09, 0.152, 0.115, 0.1432, 0.1696, 0.1528,
                  0.126, 0.1126, 0.138, 0.194, 0.109, ]
    ulimits = [0.175, 0.17, 0.17, 0.088, 0.187, 0.1643, 0.0876]
    
    # L band D_R
    detections = [0.1553, 0.1655, 0.263, 0.0465, 0.148, 0.195, 0.125, 0.112, 0.208]
    ulimits = [0.0838, 0.075]

    # Preparing distributions
    distributions_data = ((vec_lnlognorm, [0.0, 0.25]),
                          (vec_lngenbeta,[2.0, 3.0, 0.0, 0.1]),
                          (vec_lnunif,[-math.pi,math.pi]),
                          (vec_lngenbeta, [3.0, 8.0, 0.0, 0.2]),
                          (vec_lnunif, [-math.pi,math.pi]),
                          (vec_lnunif, [-math.pi,math.pi]))
    distributions = list()
    # Setting up emcee
    nwalkers = 200
    ndim = 1 
    p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))
    for (func, args) in distributions_data:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=args,
                                        bcast=True)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        distributions.append(sampler.flatchain[:,0][::2].copy())

    # Sampling posterior density of ``d``

    # Prepairing callable posterior density
    lnpost = LnPost(detections, ulimits, distributions, lnpr=lnunif,\
                    args=[0., 1.])
    
    # Using affine-invariant MCMC
    nwalkers = 400
    ndim = 1
    p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, threads=4)
    pos, prob, state = sampler.run_mcmc(p0, 50)
    sampler.reset()

    sampler.run_mcmc(pos, 200)
    d = sampler.flatchain[:,0][::2].copy()

    hist_d, edges_d = histogram(d, normed=True)
    lower_d = np.resize(edges_d, len(edges_d) - 1)
    bar(lower_d, hist_d, width=np.diff(lower_d)[0], color='g', alpha=0.5, label="D")

    
    # Using PT
    ntemps = 20
    nwalkers = 100
    ndim = 1
    logl = logl
    logp = LnPost(detections, ulimits, distributions, lnpr=lnunif,\
                    args=[0., 1.])
    # Initializing sampler
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=4)
    # Generating starting values
    p0 = np.random.uniform(low=0.05, high=0.2, size=(ntemps, nwalkers, ndim))
    # Burn-in
    for p, lnprob, lnlike in sampler.sample(p0, iterations=25):
        pass
    
    sampler.reset()
    
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                           lnlike0=lnlike,
                                           iterations=200):
        pass

    assert sampler.chain.shape == (ntemps, nwalkers, 200, ndim)

    # Chain has shape (ntemps, nwalkers, nsteps, ndim)
    # Zero temperature mean:
    mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

    # Longest autocorrelation length (over any temperature)
    max_acl = np.max(sampler.acor)
    
    
    # Using MH MCMC
    p0 = [0.5]
    sampler_mh = emcee.MHSampler(cov=[[0.05]], dim=1, lnprobfn=lnpost)
    for results in sampler_mh.sample(p0, iterations=1000):
        pass