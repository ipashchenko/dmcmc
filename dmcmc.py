#!/usr/bin python
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import special
import emcee


# TODO: make data dictionary ``distributions`` with keys: r, M, fi_M, D_2,
# fi_2, fi_1
class LnPost(object):
    """
    Class that represents posterior density.
    """
    
    def __init__(self, detections, ulimits, distributions, lnpr=None):
        """
        Parameters:
        
            detections - values of cross-to-parallel hands ratios for the
                         detected cross-hands,
            
            ulimits - upper limits on cross-to-parallel hands ratios
        
            distributions - np.array (5, N) or list of 5 arrays (N,).
                            Distributions of (|r|, |M|, fi_M, |D_2|, fi_2, fi_1),
            
            lnpr - callable prior on amplitude of the D-term.
        """
        
        self.detections = detections
        self.ulimits = ulimits
        self.distributions = distributions
        
    def __call__(self, d):
        """
        Returns posterior probability of d.
        """
        
        # TODO: using ``detections`` and ``ulimits`` find probability of them
        # being generated from cross-to-parallel hands distribution generated
        # by distributions of other parameters and given amplitude of the
        # D-term ``d``.
        
        d_distribution = self.model(d)
        lnlks_detections = [self.lnprob(detection, d_distribution) for
                            detection in self.detections]
        lnlks_ulimits = [self.lnprob(ulimit, d_distribution, type='u') for
                         ulimit in self.ulimits]
        lnlkss = lnprobs_detections.extend(lnprobs_ulimits)
        lnlk = lnprobs.sum()
        
        return lnlk + self.lnpr(d)
        
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

        result = self.data[1] * np.exp(1j * self.data[2]) + self.data[3] *\
                 np.exp(1j * self.data[4]) + d * np.exp(1j * self.data[5])

        return self.data[0] * np.sqrt((result * result.conjugate()).real)
        
    def lnprob(self, x, distribution, type=None):
        """
        Method that given some value ``x`` and sample from some distribution
        (container of values) returns natural logarithm of likelihood of
        ``x`` being generated from given distribution.
        
        Parameters:
        
            x - value of interest,
            
            distribution - [container] - sample from distribution that is checked,
            
            type [None, 'u', 'l'] - type of value ``x`` (detection, upper or
            lower limit).
            
        Output:
        
        Natural logarithm of likelihood of ``x`` being generated from
        sample's distribution.
        """
        
    def lnpr(self, d):
        """
        Prior on D-term amplitude.
        """
        
        return self._lnpr(d)
        
        
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


if __name__ == '__main__()':
    
    distributions_data = ((vec_lnlognorm, [0.0, 0.25]),
                          (vec_lngenbeta,[2.0, 3.0, 0.0, 0.1]),
                          (vec_lnunif,[-math.pi,math.pi]),
                          (vec_lngenbeta, [3.0, 8.0, 0.0, 0.2]),
                          (vec_lnunif, [-math.pi,math.pi]),
                          (vec_lnunif, [-math.pi,math.pi])) 
    distributions = list()
    # Setting up emcee
    nwalkers = 250    
    ndim = 1
    p0 = [np.random.rand(ndim)/10. for i in xrange(nwalkers)]
    # ``func`` - callable, ``args`` - list of it's arguments
    for (func, args) in distributions_data:    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=args,
                                        bcast=True)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        distributions.append(sampler.flatchain[:,0][::2].copy())


 