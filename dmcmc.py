#!/usr/bin python
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import special
from astroML.density_estimation.histtools import histogram
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
        lnlks_detections = self.lnprob(detections, d_distribution)
        lnlks_ulimits = self.lnprob(ulimits, d_distribution, type='u')
        lnlks = lnprobs_detections.extend(lnprobs_ulimits)
        lnlk = lnlks.sum()
        
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
        
        pass
    
        try:
            hist_d, edges_d = histogram(distribution, bins='knuth', normed=True)
        except IndexError:
	        hist_d = None
	        edges_d = None

        if hist_d is not None:
			# Resize edges by 1 from right
            lower_d = np.resize(edges_d, len(edges_d) - 1)

		knuth_width = np.diff(lower_d)[0]
		probs = np.zeros(len(xs))
				
		if kind is None:
			for i in range(len(probs)):
				probs[i] = hist_d[((lower_d[:, np.newaxis] - xs).T > 0)[i]][0] *\
						   knuth_width
	
		elif kind is 'u':
			for i in range(len(probs)):
				probs[i] = knuth_width * hist_d[np.where(((lower_d[:, np.newaxis]\
								                           - xs).T)[i] < 0)].sum()
								
		elif kind is 'l':
			raise NotImplementedError('Please, implement lower limits!')
					
		else:
			raise Exception('``kind`` parameter must be ``None``, ``u`` or ``l``.')
	
		return np.log(probs).sum()

        
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


 