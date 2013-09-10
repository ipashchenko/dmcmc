#!/usr/bin python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/ilya/work/emcee')
import math
import emcee
import numpy as np
from scipy import special
from knuth_hist import histogram
from matplotlib.pyplot import bar, text, xlabel, ylabel, axvline, rc


class LnPost(object):
    """
    Class that represents posterior density of given amplitude of D-.
    Using ``detections`` and ``ulimits`` parameters find probability of them
    being generated from cross-to-parallel hands distribution generated
    by distributions of other model parameters ``distributions`` and given
    amplitude of the D-term ``d``.

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

    def __call__(self, p):
        """
        Returns lnlikelihood of detections and ulimits being generated from
        distributions generated by model with given models distributions and given
        p.
        """

        ratio_distribution = self.model(p)
        print "Done modeling ratio distibution!"
        lnlks_detections = self.lnprob(self.detections, ratio_distribution)
        print "Ln of prob. for detections is : " + str(lnlks_detections)
        lnlks_ulimits = self.lnprob(self.ulimits, ratio_distribution, kind='u')
        print "Ln of prob. for ulimits is : " + str(lnlks_ulimits)

        lnlks = lnlks_detections + lnlks_ulimits
        lnlk = lnlks.sum()

        return lnlk

    def model(self, p):
        """
        Method that given amplitude of the D-term returns distribution of
        cross-to-parallel hands ratios.

        Parameters:

             p - len(p) = 2 - amplitude of the D-term, phase of the D-term

        Output:

            np.array of N values of cross-hand ratios.
        """

        data = self.distributions

        result = data[1] * np.exp(1j * data[2]) + data[3] * np.exp(1j * data[4])\
                 + p[0] * np.exp(1j * p[1])

        return data[0] * np.sqrt((result * result.conjugate()).real)

    def model_vectorized(self, ps):
        """
        Method that given amplitudes of the D-term returns distributions of
        cross-to-parallel hands ratios.

        Parameters:

            ps - len(ps) = (2,N) ensemble of amplitudes&phases of the D-term,

        Output:

            np.array of (N, len(self.distributions[i]) values of cross-hand ratios.
        """

        return self.model(p[..., np.newaxis])

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


def logp(x):
    """
    logPrior used in PT.
    """

    return lnunif(x, 0., 1.)


def percent(xs, perc=None):
    """
    Find ``perc`` % in sorted container xs.
    """

    xs_ = sorted(xs)
    indx = int(math.ceil(len(xs) * perc / 100.))

    return xs_[indx]


if __name__ == '__main__()':

    # C band D_L
    detections = [0.143, 0.231, 0.077, 0.09, 0.152, 0.115, 0.1432, 0.1696, 0.1528,
                  0.126, 0.1126, 0.138, 0.194, 0.109, ]
    ulimits = [0.175, 0.17, 0.17, 0.088, 0.187, 0.1643, 0.0876]

    # L band D_R
    #detections = [0.1553, 0.1655, 0.263, 0.0465, 0.148, 0.195, 0.125, 0.112, 0.208,
    #               0.326]
    #ulimits = [0.0838, 0.075]

    # Preparing distributions
    distributions_data = ((vec_lnlognorm, [0.0, 0.25]),
                          (vec_lngenbeta,[2.0, 3.0, 0.0, 0.1]),
                          (vec_lnunif,[-math.pi,math.pi]),
                          (vec_lngenbeta, [3.0, 8.0, 0.0, 0.2]),
                          (vec_lnunif, [-math.pi,math.pi]))
    distributions = list()
    # Setting up emcee
    nwalkers = 200
    ndim = 2
    p0 = [np.random.rand(ndim)/10. for i in xrange(nwalkers)]
    for (func, args) in distributions_data:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=args,
                                        bcast=True)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        # Using only 10000 points for specifying distributions
        distributions.append(sampler.flatchain[:,0][::20].copy())

    # Sampling posterior density of ``d``

    ## Prepairing callable posterior density
    #lnpost = LnPost(detections, ulimits, distributions, lnpr=lnunif,\
                    #args=[0., 1.])

    ## Using affine-invariant MCMC
    #nwalkers = 400
    #ndim = 1
    #p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, threads=4)
    #pos, prob, state = sampler.run_mcmc(p0, 50)
    #sampler.reset()

    #sampler.run_mcmc(pos, 200)
    #d = sampler.flatchain[:,0][::2].copy()

    #hist_d, edges_d = histogram(d, normed=True)
    #lower_d = np.resize(edges_d, len(edges_d) - 1)
    #bar(lower_d, hist_d, width=np.diff(lower_d)[0], color='g', alpha=0.5)


    # Using PT
    ntemps = 20
    nwalkers = 100
    ndim = 2
    # logLikelihood
    logl = LnLike(detections, ulimits, distributions)
    # logprior
    logp = logp
    # Initializing sampler
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=12)
    # Generating starting values
    p0_0 = np.random.uniform(low=0.05, high=0.2, size=(ntemps, nwalkers, 1))
    p0_1 = np.random.uniform(low=-math.pi, high=math.pi, size=(ntemps, nwalkers, 1))
    p0 = np.concatenate((p0_0, p0_1), axis=2)
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

    # shortcut for zero-temperature chain
    d = sampler.chain[0,:,::10].T.reshape(2000)
    hist_d, edges_d = histogram(d, normed=True)
    lower_d = np.resize(edges_d, len(edges_d) - 1)
    bar(lower_d, hist_d, width=np.diff(lower_d)[0], color='g', alpha=0.5)
    font = {'family': 'Droid Sans', 'weight': 'normal', 'size': 18}
    xlabel(ur"$C$-диапазон $D_{L}$")
    ylabel(ur"плотность вероятности")
    #text(0.06, 50, r'$\mu$  = $0.143$, 95$\%$ интервал: [0.011, 0.174]', fontdict={'family':
    #                                                            'Droid Sans',\
    #                                                            'weight': 'normal',\
    #                                                            'size': 18})
    axvline(x=0.147, linewidth=2, color='r')
    axvline(x=0.115, color='r')
    axvline(x=0.172, color='r')

    # Predictive density analysis
    predictive_ratios = logl.model_vectorized(d)
    # dim = (len(d), len(distributions[i]))
    simulated_datas = [np.random.choice(ratio, size=len(detections)) for ratio
                      in predictive_ratios]
    # Lists of some statistics
    simulated_means = map(np.mean, simulated_datas)
    simulated_maxs = map(np.max, simulated_datas)
    simulated_mins = map(np.min, simulated_datas)

    # Histograms of statistics for simalated data
    hist_means, edges_means = histogram(simulated_means, normed=True)
    lower_means = np.resize(edges_means, len(edges_means) - 1)
    bar(lower_means, hist_means, width=np.diff(lower_means)[0], color='g', alpha=0.5)

    hist_maxs, edges_maxs = histogram(simulated_maxs, normed=True)
    lower_maxs = np.resize(edges_maxs, len(edges_maxs) - 1)
    bar(lower_maxs, hist_maxs, width=np.diff(lower_maxs)[0], color='r', alpha=0.5)

    hist_mins, edges_mins = histogram(simulated_mins, normed=True)
    lower_mins = np.resize(edges_mins, len(edges_mins) - 1)
    bar(lower_mins, hist_mins, width=np.diff(lower_mins)[0], color='b', alpha=0.5)

    # Draw realized data's statistic
    axvline(x=np.mean(detections), linewidth=2, color='g')
    axvline(x=np.min(detections), linewidth=2, color='b')
    axvline(x=np.max(detections), linewidth=2, color='r')

    # Draw 5% & 95% borders
    axvline(x=percent(simulated_means, proc=5.0), color='g')
    axvline(x=percent(simulated_means, proc=95.0), color='g')
    axvline(x=percent(simulated_maxs, proc=5.0), color='r')
    axvline(x=percent(simulated_maxs, proc=95.0), color='r')
    axvline(x=percent(simulated_mins, proc=5.0), color='b')
    axvline(x=percent(simulated_mins, proc=95.0), color='b')

    # Using MH MCMC
    p0 = [0.5]
    sampler_mh = emcee.MHSampler(cov=[[0.05]], dim=1, lnprobfn=lnpost)
    for results in sampler_mh.sample(p0, iterations=1000):
        pass