#!/usr/bin python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/ilya/work/emcee')
import math
#import emcee
import numpy as np
import numdifftools as nd
from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
from scipy.stats.kde import gaussian_kde
#from knuth_hist import histogram
#from matplotlib.pyplot import bar, text, xlabel, ylabel, axvline, rc


# TODO: add power analysis to paper. Use resampling of known detections to find
# sample size that is enough for 95%HDI be less then 0.03. Try firstly find
# posterior of D_RA, then create many data seta from this posterior with reliable
# volume each end finally find the proportion of cases where ROPE (defined by
# D_RA posterior that generated all data sets) in 95% HDI.
# TODO: Likelihood is pretty gaussian => can use Laplace appr. to evidence. How
# often during power analysis
class LnPost(object):
    """
    Class that represents posterior density of parameters.
    Using ``detections`` and ``ulimits`` parameters find probability of them
    being generated from cross-to-parallel hands distribution generated
    by distributions of model parameters presented as ``distributions`` and
    other model parameters.
    """

    def __init__(self, detections, ulimits, distributions, lnpr=None,
                 args=None):
        self._lnpr = lnpr
        self.args = args
        self._lnlike = LnLike(detections, ulimits, distributions)

    def lnpr(self, p):
        return self._lnpr(p, *self.args)

    def lnlike(self, p):
        return self._lnlike.__call__(p)

    def __call__(self, p):
        return self.lnlike(p) + self.lnpr(p)


class LnLike(object):
    """
    Class representing Likelihood function.
    """

    def __init__(self, detections, ulimits, distributions):
        """
        Parameters:

            detections - values of cross-to-parallel hands ratios for the
                         detected cross-hands,

            ulimits - upper limits on cross-to-parallel hands ratios

            distributions - distributions of (|r|, |M|, fi_M, |D_2|, fi_2, fi_1).
                            or list of (callable, args, kwargs,)
            size - size of model distributions that will be used for estimating
                pdf of cross-to-parallel hands ratios for given other model
                parameters. If it is None then distributions is treated like
                several data samples taken from each distribution. If it is
                set then distribution is treated like list of tuples with
                callables and arguments.
        """

        self.detections = detections
        self.ulimits = ulimits
        self.distributions = distributions

    def __call__(self, p):
        """
        Returns lnlikelihood of detections and ulimits being generated from
        distributions generated by model with given models distributions and given
        d.
        """

        ratio_distribution = self.model(p)
        lnlk_detections = self.lnprob(self.detections, ratio_distribution)
        lnlk_ulimits = self.lnprob(self.ulimits, ratio_distribution, kind='u')

        lnlk = lnlk_detections + lnlk_ulimits

        return lnlk

    def model(self, p):
        """
        Method that given parameter vector returns pdf of cross-to-parallel
        hands ratios.

        Parameters:

            p - single walker

        Output:

            np.array of N values of cross-hand ratios.
        """

        data = self.distributions

        try:
            result = data[1][0, :] * np.exp(1j * data[2]) + data[3] *\
                     np.exp(1j * data[4]) + p * np.exp(1j * data[5])
        except IndexError:
            result = data[1] * np.exp(1j * data[2]) + data[3] * np.exp(1j *\
                     data[4]) + p * np.exp(1j * data[5])

        return data[0] * np.sqrt((result * np.conj(result)).real)

    def model_vectorized(self, p):
        """
        Method that given ensemble of walkers returns distributions of
        cross-to-parallel hands ratios.

        Parameters:

            p - ensemble of walkers

        Output:

            np.array of (N, len(self.distributions[i]) values of
            cross-to-parallel hand ratios.
        """

        return self.model(p[:, np.newaxis])

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
            probs = kde(xs)
        elif kind is 'u':
            for i in range(len(probs)):
                # For ratio distribution minimum is zero!
                probs[i] = kde.integrate_box_1d(0, xs[i])
        elif kind is 'l':
            raise NotImplementedError('Please, implement lower limits!')
        else:
            raise Exception('``kind`` parameter must be ``None``, ``u``\
                            or ``l``.')

        result = np.log(probs).sum()

        return result


class _distribution_wrapper(object):
    """
    This is a hack to make the distribution function picklable when ``args``
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
                print("mmcmc: Exception while calling your distribution callable:")
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


def fn_distributions(mmin=None, mmax=None, ma=None, mb=None, dmin=None,
                     dmax=None, da=3, db=8, rmean=0, rsigma=0.25, size=10000):

    return ((np.random.lognormal, list(), {'mean': rmean, 'sigma': rsigma,
                'size': size},),
            (genbeta, [mmin, mmax, ma, mb], {'size': size},),
            (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi,
                'size': size},),
            (genbeta, [dmin, dmax, da, db], {'size': size},),
            (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi,
                'size': size},),
            (np.random.uniform, list(), {'low': -math.pi, 'high': math.pi,
                'size': size},))


def laplace_logevidence(lnpost):
    """
    Function that given ln of posterior pdf returns ln of evidence using Laplace
    approximation.
    """

    # Find MAP
    result = optimize.minimize_scalar(lambda x: -lnpost(x), bounds=[0.01, 0.25],
                                     method='Bounded')
    x_map = result['x']
    # Find Hessian at MAP
    hess = nd.Hessdiag(lnpost)
    hess_map = hess(x_map)

    try:
        M = len(x_map)
    except TypeError:
        M = 1
    print x_map
    print lnpost(x_map)
    print M
    print hess_map[0]
    return lnpost(x_map) + 0.5 * M * math.log(2 * math.pi) -\
           0.5 * np.log(abs(hess_map[0]))


def find_laplace_logZ(detections, ulimits, mmin=None, mmax=None, ma=None, mb=None,
                   dmin=None, dmax=None, da=3, db=8, rmean=0, rsigma=0.25,
                   size=10000, lnpr=lnunif, args=[0., 1.], epsrel=0.001):

    distributions = [_distribution_wrapper(d[0], d[1], d[2])() for d in
                     fn_distributions(mmin=mmin, mmax=mmax, ma=ma, mb=mb,
                                      dmin=dmin, dmax=dmax, size=size, da=da,
                                      db=db)]
    lnpost = LnPost(detections, ulimits, distributions, lnpr=lnpr, args=args)

    return laplace_logevidence(lnpost)


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    """
    Highest density interval of sample.
    """

    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    return hdi_min, hdi_max


def hdi_of_plot(y, x, fn, cred_mass=0.95):
    """
    Highest density interval for (x, y). Currently works only for one-mode
    densities.
    """

    lvl = 0.50

    def get_cred(x, y, fn, lvl):

        y = np.array(y)
        # Find max of y's
        ymax = max(y)
        # Select some level
        y0 = lvl * ymax
        x1 = x[np.where(y - y0 > 0)][0]
        x2 = x[np.where(y - y0 > 0)][-1]
        Z = integrate.quad(fn, 0, 1, full_output=0, epsrel=0.001)[0]
        cred = integrate.quad(fn, x1, x2)[0] / Z

        return cred, x1, x2

    def d_cred(lvl, x, y, fn, cred_mass):

        cred = get_cred(x, y, fn, lvl)[0]
        return cred_mass - cred

    lvl_star = optimize.brentq(d_cred, 0.01, 0.99, args=(x, y, fn, cred_mass))

    cred, x1, x2 = get_cred(x, y, fn, lvl_star)
    print "Got x1 = " + str(x1) + ' & x2 = ' + str(x2) + ' with cred: ' +\
           str(cred)

    while(cred < cred_mass):

        if cred_mass - cred > 0.1:
            lvl -= 0.1
            print "Going down on 0.1 in lvl"
        elif 0.05 < cred_mass - cred < 0.1:
            lvl -= 0.05
            print "Going down on 0.05 in lvl"
        if cred_mass - cred < 0.05:
            lvl -= 0.01
            print "Going down on 0.01 in lvl"

        print "lvl = " + str(lvl)
        cred, x1, x2 = get_cred(x, y, fn, lvl)
        print "Got x1 = " + str(x1) + ' & x2 = ' + str(x2) + ' with cred: ' +\
            str(cred)

    return cred, x1, x2


def find_cred_interval_1d(post, cred_mass=0.95, a=0, b=1,
                          xmax_interval=[0.01, 0.25]):
    """
    Find 95% credibility interval for unimodal posterior prob. density ``post``.
    """
    # MAP
    xmap = optimize.minimize_scalar(lambda x: -post(x), bounds=xmax_interval,
                                     method='Bounded')['x']
    pmap = post(xmap)

    def get_cred(lvl, post, xmap, a=a, b=b):
        pmap = post(xmap)
        p0 = pmap * lvl
        x1 = optimize.brentq(lambda x: post(x) - p0, a, xmap)
        x2 = optimize.brentq(lambda x: post(x) - p0, xmap, b)
        Z = integrate.quad(post, a, b, full_output=0, epsrel=0.001)[0]
        cred = integrate.quad(post, x1, x2)[0] / Z
        return cred, x1, x2

    def d_cred(lvl, post, pmap, a=a, b=b, cred_mass=cred_mass):
        cred = get_cred(lvl, post, xmap, a=a, b=b)[0]
        return cred_mass - cred

    lvl_star = optimize.brentq(d_cred, 0.01, 0.75, args=(post, pmap, a, b,
                                                         cred_mass))

    return get_cred(lvl_star, post, xmap, a=a, b=b)[1:]


def find_grid_logZ(detections, ulimits, mmin=None, mmax=None, ma=None, mb=None,
                   dmin=None, dmax=None, da=3, db=8, rmean=0, rsigma=0.25,
                   size=10000, lnpr=lnunif, args=[0., 1.], epsrel=0.001):

    distributions = [_distribution_wrapper(d[0], d[1], d[2])() for d in
                     fn_distributions(mmin=mmin, mmax=mmax, ma=ma, mb=mb,
                                      dmin=dmin, dmax=dmax, size=size, da=da,
                                      db=db)]
    lnpost = LnPost(detections, ulimits, distributions, lnpr=lnpr, args=args)

    def fn(x):
        return math.exp(lnpost(x))

    result = integrate.quad(fn, 0, 1, full_output=0, epsrel=epsrel)

    return math.log(result[0])


def model_analysis(detections, ulimits, mmin=None, mmax=None, ma=None, mb=None,
                   dmin=None, dmax=None, da=None, db=None, a=0., b=1.,
                   xmap_interval=[0.01, 0.25]):

    for kwarg in [mmin, mmax, ma, mb, dmin, dmax, da, db, a, b, xmap_interval]:
        assert(kwarg is not None)

    # dictionary of model parameters
    model = dict()
    model.update({"mmin": mmin, "mmax": mmax, "ma": ma, "mb": mb, "dmin": dmin,
                  "dmax": dmax, "da": da, "db": db})

    # Preparing distributions for current model
    print "Creating model distributions"
    distributions = [_distribution_wrapper(d[0], d[1], d[2])() for d in
                     fn_distributions(mmin=mmin, mmax=mmax, ma=ma, mb=mb,
                                      dmin=dmin, dmax=dmax, size=10000, da=da,
                                      db=db)]

    # Initialize LnPosterior of current model
    print "Initializing Posterior pdf"
    lnpost = LnPost(detections, ulimits, distributions, lnpr=lnunif,
                    args=[a, b])

    # Find MAP D_RA and probability of it
    print "Calculating MAPs"
    xmap = optimize.minimize_scalar(lambda x: -math.exp(lnpost(x)),
                                    bounds=xmap_interval,
                                    method='Bounded')['x']
    pmap = math.exp(lnpost((xmap)))

    # Find normalizing constant (evidence)
    print "Calculating evidence of model"
    Z = integrate.quad(lambda x: math.exp(lnpost(x)), a, b, full_output=0, epsrel=0.001)[0]
    # Add it to model dictionary
    model.update({"lnZ": math.log(Z)})

    # Create plausible range of bulk posterior for D_RA for fitting
    ds_ra = np.arange(250) / 1000.
    # Calculate near mode
    print "Calculating probabilities for fitting with gaussian"
    post_probs = [math.exp(lnpost(d)) / Z for d in ds_ra]

    # Gaussian for fitting the posterior pdf
    def gauss(x, *p):
        amp, mu, sigma = p
        return amp * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    # Fit with gaussian
    print "Fitting posterior pdf with gaussian"
    p = optimize.curve_fit(gauss, ds_ra, post_probs, p0=[pmap, xmap, 0.1])
    mu, sigma = p[0][1:]
    sigma = abs(sigma)

    # Sample from gaussian pdf of D_RA 1000 D_RA values
    print "Generating values of D_RA from posterior"
    ds_post = np.random.normal(loc=mu, scale=sigma, size=1000)

    # Use model to get predicted ratios
    # dim = (1000, len(distributions[i]=10000))
    print "Creating replicated data sets"
    distr_ratios_replicas = lnpost._lnlike.model_vectorized(ds_post)
    detections_replicas = [np.random.choice(distr_ratio, size=len(detections))
                           for distr_ratio in distr_ratios_replicas]

    # Now check plausibility of model using statistics of original data &
    # replicas:
    print "Calculating statistics for replicated data sets"
    simulated_means = map(np.mean, detections_replicas)
    simulated_maxs = map(np.max, detections_replicas)
    simulated_mins = map(np.min, detections_replicas)

    # Check what quantils are corresponds to real data
    print "Comparing replicas statistics with data"
    percentiles = dict()
    percentiles.update({"min": stats.percentileofscore(simulated_mins,
                                                       np.min(detections))})
    percentiles.update({"max": stats.percentileofscore(simulated_maxs,
                                                       np.max(detections))})
    percentiles.update({"mean": stats.percentileofscore(simulated_means,
                                                        np.mean(detections))})
    model.update({"perc": percentiles})

    return model


def power_analysis(detections, ulimits, size=None, a=0., b=1., mmin=None,
                   mmax=None, ma=None, mb=None, dmin=None, dmax=None, da=None,
                   db=None):

    # Create dataset with predefined size by resampling from original data pdf
    print "Creating data sample with size = " + str(size)
    kde = gaussian_kde(detections)
    newdets = kde.resample(size=size)[0]
    print newdets

    # Preparing distributions for current model
    print "Creating model distributions"
    distributions = [_distribution_wrapper(d[0], d[1], d[2])() for d in
                     fn_distributions(mmin=mmin, mmax=mmax, ma=ma, mb=mb,
                                      dmin=dmin, dmax=dmax, size=10000, da=da,
                                      db=db)]

    print "Initializing Posterior pdf"
    lnpost = LnPost(newdets, ulimits, distributions, lnpr=lnunif, args=[a, b])

    print "Finding ranges of 95% HDI of the posterior"
    x1, x2 = find_cred_interval_1d(lambda x: math.exp(lnpost(x)), cred_mass=0.95,
                                   a=0, b=1, xmax_interval=[0.01, 0.25])

    print x1, x2

    return x2 - x1


if __name__ == '__main__()':

    # C band D_L
    detections = [0.143, 0.231, 0.077, 0.09, 0.152, 0.115, 0.1432, 0.1696, 0.1528,
                  0.126, 0.1126, 0.138, 0.194, 0.109, 0.101]
    ulimits = [0.175, 0.17, 0.17, 0.088, 0.187, 0.1643, 0.0876, 0.123, 0.77,
               0.057, 0.155]

    # Create data set from hypothetical parameters
    # Preparing distributions
    distributions = [_distribution_wrapper(d[0], d[1], d[2])() for d in
                     fn_distributions(mmin=0, mmax=0.1, ma=1, mb=8, dmin=0.01,
                                      dmax=0.15, size=10000, da=3, db=8)]

    # Prepare sample of D_RA values from hypothetical distribution:
    d_hypo = genbeta(0.09, 0.11, 5, 5, size=500)

    # Initialize LnPost class - we need methods of it's objects
    lnpost = LnPost(detections, ulimits, distributions, lnpr=vec_lngenbeta,
                    args=[5, 5, 0.08, 0.12])

    # 500 samples with 10000 data points each
    predictive_ratios = lnpost._lnlike.model_vectorized(d_hypo)
    # Create 500 data samples with 100 data points in each
    simulated_datas = [np.random.choice(ratio, size=100) for ratio in
                       predictive_ratios]

    # Or use kde
    kde = gaussian_kde(detections)
    newdets = kde.resample(size=200)[0]

    # Find MAP
    lnpost = LnPost(newdets, ulimits, distributions, lnpr=lnunif,
                    args=[0., 1.])
    x_map = optimize.minimize_scalar(lambda x: -lnpost(x), bounds=[0.01, 0.25],
                                     method='Bounded')['x']

    probs = [math.exp(lnpost(p)) for p in np.arange(250) / 1000.]
