#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import os
import numpy as np
import scipy
import scipy.linalg
import multiprocessing
import time

class AdaptiveCovarianceMCMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Creates a chain of samples from a target distribution, using the adaptive
    covariance routine described in [1].

    The algorithm starts out as basic MCMC, but after a certain number of
    iterations starts tuning the covariance matrix, so that the acceptance rate
    of the MCMC steps converges to a user specified rate.

    By default, the method will run for ``2000 * d`` iterations, where ``d`` is
    the dimension of the used :class:`LogLikelihood`. Of these, the first 25%
    will run without adaptation, and the first 50% will be discarded as
    burn-in. These numbers can be modified using the methods

    #TODO

    [1] Uncertainty and variability in models of the cardiac action potential:
    Can we build trustworthy models?
    Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan, Clayton,
    Mirams (2015) Journal of Molecular and Cellular Cardiology
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(AdaptiveCovarianceMCMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Target acceptance rate
        self._acceptance_target = 0.25

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Number of iterations before adapation
        self._adaptation = int(0.25 * self._iterations)

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

    def acceptance_rate(self):
        """
        Returns the target acceptance rate that will be used in the next run.
        """
        return self._acceptance_target

    def burn_in(self):
        """
        Returns the number of iterations that will be discarded as burn-in in
        the next run.
        """
        return self._burn_in

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run, including the non-adaptive and burn-in iterations.
        """
        return self._iterations

    def non_adaptive_iterations(self):
        """
        Returns the number of initial, non-adaptive iterations that will be
        performed in the next run.
        """
        return self._adaptation

    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Report the current settings
        if self._verbose:
            print('Running adaptive covariance MCMC')
            print('Target acceptance rate: ' + str(self._acceptance_target))
            print('Total number of iterations: ' + str(self._iterations))
            print(
                'Number of iterations before adapation: '
                + str(self._adaptation))
            print(
                'Number of iterations to discard as burn-in: '
                + str(self._burn_in))
            print('Storing one sample per ' + str(self._thinning_rate))

        # Problem dimension
        d = self._dimension

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError(
                'Suggested starting position has a non-finite log-likelihood.')

        # Chain of stored samples
        stored = int((self._iterations - self._burn_in) / self._thinning_rate)
        chain = np.zeros((stored, d))

        # Initial acceptance rate (value doesn't matter)
        loga = 0
        acceptance = 0

        # Save to file
        if self._save:
            savedir = os.path.join(self._save_dir, 'adaptive-mcmc.txt')
            with open(savedir, 'w') as outfile:
                outfile.write('# Adaptive covariance matrix\n')
                outfile.write('# Thinned by taking saving only every '
                              + str(thinning) + '-th accepted state\n')
                outfile.write('# 1-'+str(d)+'. parameters, '
                              +str(1+d)+'. logLikelihood, '
                              +str(2+d)+'. acceptance rate, '
                              +str(3+d)+'. loga, '
                              +str(4+1*d)+'-'+str(4+2*d-1)+'. mean estimate, '
                              +str(4+2*d)+'. real time taken (s), '
                              +str(5+2*d)+'. iteration\n')
            H = [np.concatenate((np.copy(current),
                                [current_log_likelihood, acceptance, loga],
                                mu,
                                [time.time()-start,0]
                                ))]

        # Go!
        for i in range(self._iterations):
            # Propose new point
            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            proposed = np.random.multivariate_normal(
                current, np.exp(loga) * sigma)

            # Check if the point can be accepted
            accepted = 0
            proposed_log_likelihood = self._log_likelihood(proposed)
            if np.isfinite(proposed_log_likelihood):
                u = np.log(np.random.rand())
                if u < proposed_log_likelihood - current_log_likelihood:
                    accepted = 1
                    current = proposed
                    current_log_likelihood = proposed_log_likelihood

            # Adapt covariance matrix
            if i >= self._adaptation:
                gamma = (i - self._adaptation + 2) ** -0.6
                mu = (1 - gamma) * mu + gamma * current
                loga += gamma * (accepted - self._acceptance_target)
                dsigm = np.reshape(current - mu, (d, 1))
                sigma = (1 - gamma) * sigma + gamma * np.dot(dsigm, dsigm.T)

            # Update acceptance rate
            acceptance = (i * acceptance + float(accepted)) / (i + 1)

            # Add point to chain
            ilog = i - self._burn_in
            if ilog >= 0 and ilog % self._thinning_rate == 0:
                chain[ilog // self._thinning_rate, :] = current
                if self._save:
                    # Only save after burn-in
                    H.append(np.concatenate((
                        np.copy(current),
                        [current_log_likelihood, acceptance, loga],
                        mu,
                        [time.time()-start,i]
                        )))

            # Report
            if self._verbose and i % 50 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  In burn-in: ' + str(i < self._burn_in))
                print('  Adapting: ' + str(i >= self._adaptation))
                print('  Acceptance rate: ' + str(acceptance))
            # Save
            if self._save and i % 500 == 0:
                with open(savedir, 'a') as outfile:
                    np.savetxt(outfile, H)
                H = []

        # Check that chain fully filled
        if ilog // self._thinning_rate != len(chain) - 1:
            raise Exception('Unexpected error: Chain not fully generated.')

        # Return generated chain
        return chain

    def set_acceptance_rate(self, rate):
        """
        Sets the target acceptance rate for the next run.
        """
        rate = float(rate)
        if rate <= 0:
            raise ValueError('Target acceptance rate must be greater than 0.')
        elif rate > 1:
            raise ValueError('Target acceptance rate cannot exceed 1.')
        self._acceptance_target = rate

    def set_burn_in(self, burn_in):
        """
        Sets the number of iterations to discard as burn-in in the next run.
        """
        burn_in = int(burn_in)
        if burn_in < 0:
            raise ValueError('Burn-in rate cannot be negative.')
        self._burn_in = burn_in

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including burn-in and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_non_adaptive_iterations(self, adaptation):
        """
        Sets the number of iterations to perform before using adapation in the
        next run.
        """
        adaptation = int(adaptation)
        if adaptation < 0:
            raise ValueError('Adaptation cannot be negative.')
        self._adaptation = adaptation

    def set_thinning_rate(self, thinning):
        """
        Sets the thinning rate. With a thinning rate of *n*, only every *n-th*
        sample will be stored.
        """
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be greater than zero.')
        self._thinning_rate = thinning

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate

    def load_chain(self, path_or_chain):
        """
        Load previously simulated chain for later anaylsis for
        adpative MCMC method.

        Arguments:

        ``path_or_chain``
            String: Path to saved chain.
            Array-type: Load it as current chain

        """
        d = self._dimension
        if isinstance(path_or_chain, str):
            try:
                iterations, final_loga, time_taken = np.loadtxt(path_or_chain,
                        usecols=[5+2*d-1, 3+d-1, 4+2*d-1])[-1,:]
                iterations_2 = np.loadtxt(path_or_chain,
                        usecols=[5+2*d-1])[-2]
                iterations_burn_in = np.loadtxt(path_or_chain,
                        usecols=[5+2*d-1])[1]
                self.set_iterations(int(iterations))
                self.set_thinning(int(iterations - iterations_2))
                self.set_burn_in(int(iterations_burn_in))
                self._chain = np.loadtxt(path_or_chain,
                        usecols=range(d))[1:,:]
            except:
                raise Exception('Cannot load file as chain '
                                + path_or_chain)
        elif isinstance(path_or_chain, (np.ndarray, list)):
            try:
                assert path_or_chain.shape[1] == d
                self._chain = path_or_chain
            except:
                raise Exception('Array does not match expected chain dimension')

    def chain(self):
        """
        Return chain.
        """
        return self._chain


def adaptive_covariance_mcmc(log_likelihood, x0, sigma0=None, savetofile=False):
    """
    Runs an adaptive covariance MCMC routine with the default parameters.
    """
    return AdaptiveCovarianceMCMC(log_likelihood, x0, sigma0, savetofile=savetofile).run()

