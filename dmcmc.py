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
from scipy.stats.kde import gaussian_kde


class LnPost(object):
    """
    Class that represents posterior density of given amplitude of D-.
    Using ``detections`` and ``ulimits`` parameters find probability of them
    being generated from cross-to-parallel hands distribution generated
    by distributions of other model parameters ``distributions`` and given
    amplitude of the D-term ``d``.

    """

    def __init__(self, detections, ulimits, distributions, size=None, lnpr=None, args=None):
        self._lnpr = lnpr
        self.args = args
        self._lnlike = LnLike(detections, ulimits, distributions, size=size)
        
    def lnpr(self, d):
        return self._lnpr(d, *self.args)
    
    def lnlike(self, d):
        return self._lnlike.__call__(d)
        
    def __call__(self, d):
        return self.lnlike(d) + self.lnpr(d)


class LnLike(object):
    """
    Class representing Likelihood function.
    """

    def __init__(self, detections, ulimits, distributions, size=None):
        """
        Parameters:

            detections - values of cross-to-parallel hands ratios for the
                         detected cross-hands,

            ulimits - upper limits on cross-to-parallel hands ratios

            distributions - distributions of (|r|, |M|, fi_M, |D_2|, fi_2, fi_1).
                            or list of (callable, args, kwargs,)
            size - size of model distributions that will be used for estimating pdf
                    of cross-to-parallel hands ratios for given RA D-term. If it is
                    None then distributions is treated like several data samples
                    sampled from each distribution. If it is set then distribution
                    is treated like list of tuples with callables and arguments.
        """

        self.detections = detections
        self.ulimits = ulimits
        self.size=size
        self.distributions = list()
        
        # Size of model distributions must be specified
        assert(self.size)
        
        for entry in distributions:
            entry[2].update({'size': self.size})
            self.distributions.append(_distribution_wrapper(entry[0],
                                                            entry[1], entry[2]))

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

        result = data[1]() * np.exp(1j * data[2]()) + data[3]() * np.exp(1j *
                                                                         data[4]()) + d * np.exp(1j * data[5]())

        return data[0]() * np.sqrt((result * result.conjugate()).real)

    def model_vectorized(self, d):
        """
        Method that given amplitudes of the D-term returns distributions of
        cross-to-parallel hands ratios.

        Parameters:

            d - N amplitudes of the D-term,

        Output:

            np.array of (N, len(self.distributions[i]) values of cross-hand ratios.
        """

        return self.model(d[:, np.newaxis])

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

        kde = gaussian_kde(distribution)
        probs = np.zeros(len(xs))

        if kind is None:
            for i in range(len(probs)):
                probs[i] = kde(xs).sum()
        elif kind is 'u':
            for i in range(len(probs)):
                probs[i] = kde.integrate_box_1d(min(distribution), xs[i])
        elif kind is 'l':
            raise NotImplementedError('Please, implement lower limits!')
        else:
            raise Exception('``kind`` parameter must be ``None``, ``u``\
                            or ``l``.')

        result = np.log(probs).sum()

        return result


class _distribution_wrapper(object):
    """
    This is a hack to make the distribution function pickleable when ``args``
    and ``kwargs`` are also included.
    """
    
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
            try:
                return self.f(*self.args, **self.kwargs)
            except:
                import traceback
                print("dmcmc: Exception while calling your distribution callable:")
                print("  args:", self.args)
                print("  kwargs:", self.kwargs)
                print("  exception:")
                traceback.print_exc()
                raise
            

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


def genbeta(a, b, *args, **kwargs):
    return (b - a) * np.random.beta(*args, **kwargs) + a

def prepare_sample_m(data):
    """
    Input:
    
        data = list of tuples (N, sample), where
        N - number of times source was observed,
        sample - list of frac. polarizations (or any other observable) detected in
            observations of that source before.
            
    Output:
        list of values with length sum_i(N_i) - sum of observations wuch were taken
            from each sample N times randomly.
        """

    result = list()
    for entry in data:
        subsample = np.random.choice(entry[1], size=entry[0])
        result.extend(subsample)
        
    return result
    

if __name__ == '__main__()':

    # C band D_L
    detections = [0.143, 0.231, 0.077, 0.09, 0.152, 0.115, 0.1432, 0.1696, 0.1528,
                  0.126, 0.1126, 0.138, 0.194, 0.109, 0.101]
    ulimits = [0.175, 0.17, 0.17, 0.088, 0.187, 0.1643, 0.0876, 0.123, 0.77,
               0.057, 0.155]
    
    # Preparing data for constructing frac. polarization pdf
    data = [(1, [0.069, 0.049, 0.065]), (1, [0.021, 0.084, 0.089, 0.048, 0.016,
                                             0.058, 0.055]),
            (1, [0.027, 0.014]),
            (6, [0.105, 0.117, 0.132, 0.115, 0.092, 0.066, 0.093, 0.097, 0.084,
                 0.058, 0.075, 0.093, 0.117, 0.113, 0.105, 0.105, 0.096, 0.088,
                 0.098, 0.101, 0.095, 0.081, 0.090, 0.103, 0.139, 0.138, 0.118]),
            (2, [0.010, 0.015]),
            (2, [0.017, 0.010, 0.026]),
            (1, [0.019, 0.015, 0.061, 0.032, 0.046, 0.076, 0.029, 0.039, 0.027,
                 0.014, 0.024, 0.014, 0.016, 0.037, 0.021, 0.059, 0.058, 0.060,
                 0.054, 0.064, 0.064, 0.025, 0.042, 0.009, 0.008, 0.007, 0.025,
                 0.023]),
            (1, [0.051, 0.045, 0.037, 0.036, 0.028]),
            (2, [0.031, 0.014, 0.023, 0.019, 0.023, 0.025]),
            (3, [0.009, 0.004, 0.022, 0.013])]

    datas = list()
    for i in range(1000):
        datas.extend(prepare_sample_m(data))
    # Will use this kde in model distributions
    kde = gaussian_kde(datas)

    # L band D_R
    detections = [0.1553, 0.1655, 0.263, 0.0465, 0.148, 0.195, 0.125, 0.112, 0.208,
                  0.326]
    ulimits = [0.0838, 0.075]

    # Preparing distributions
    distributions = ((np.random.lognormal, list(), {'mean': 0.0, 'sigma': 0.25}),
                      (kde.resample, list(), dict()),
                      #(genbeta, [0.0, 0.1, 2.0, 3.0], dict(),),
                      (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi}),
                      (genbeta, [0.0, 0.2, 3.0, 8.0], dict(),),
                      (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi}),
                      (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi}))
    
    # Sampling posterior density of ``d``

    lnpost = LnPost(detections, ulimits, distributions, size=10000, lnpr=lnunif,\
                    args=[0., 1.])

    # Using affine-invariant MCMC
    nwalkers = 250
    ndim = 1
    p0 = np.random.uniform(low=0.05, high=0.2, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    sampler.run_mcmc(pos, 400)
    d = sampler.flatchain[:,0][::10].copy()

    hist_d, edges_d = histogram(d, normed=True)
    lower_d = np.resize(edges_d, len(edges_d) - 1)
    bar(lower_d, hist_d, width=np.diff(lower_d)[0], color='g', alpha=0.5)


    # Using PT
    ntemps = 20
    nwalkers = 100
    ndim = 1
    # logLikelihood
    logl = LnLike(detections, ulimits, distributions, size=10000)
    # logprior
    logp = logp
    # Initializing sampler
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logl, logp)
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
    simulated_meds = map(np.median, simulated_datas)

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

    hist_meds, edges_meds = histogram(simulated_meds, normed=True)
    lower_meds = np.resize(edges_meds, len(edges_meds) - 1)
    bar(lower_meds, hist_meds, width=np.diff(lower_meds)[0], color='y', alpha=0.5)
    
    # Draw realized data's statistic
    axvline(x=np.mean(detections), linewidth=2, color='g')
    axvline(x=np.min(detections), linewidth=2, color='b')
    axvline(x=np.max(detections), linewidth=2, color='r')
    axvline(x=np.median(detections), linewidth=2, color='y')

    # Draw 5% & 95% borders
    axvline(x=percent(simulated_means, perc=5.0), color='g')
    axvline(x=percent(simulated_means, perc=95.0), color='g')
    axvline(x=percent(simulated_maxs, perc=5.0), color='r')
    axvline(x=percent(simulated_maxs, perc=95.0), color='r')
    axvline(x=percent(simulated_mins, perc=5.0), color='b')
    axvline(x=percent(simulated_mins, perc=95.0), color='b')
    axvline(x=percent(simulated_meds, perc=5.0), color='y')
    axvline(x=percent(simulated_meds, perc=95.0), color='y')

    # Using MH MCMC
    p0 = [0.1]
    sampler_mh = emcee.MHSampler(cov=[[0.05]], dim=1, lnprobfn=lnpost)
    for results in sampler_mh.sample(p0, iterations=1000):
        pass