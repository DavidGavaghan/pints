#
# Scoring functions
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ErrorMeasure(object):
    """
    Calculates some scalar measure of goodness-of-fit for a model and a data
    set, such that a smaller value means a better fit.
    """
    def __init__(self, problem):
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._dimension = problem.dimension()

    def __call__(self, x):
        raise NotImplementedError

    def dimension(self):
        """
        Returns the dimension of the space this measure is defined on.
        """
        return self._dimension


class LogLikelihoodBasedError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Inverts a log-likelihood to use it as an error.
    """
    def __init__(self, likelihood):
        if not isinstance(likelihood, pints.LogLikelihood):
            raise ValueError(
                'Argument to LikelihoodBasedError must be instance of'
                ' Likelihood')
        super(LogLikelihoodBasedError, self).__init__(likelihood._problem)
        self._likelihood = likelihood

    def __call__(self, x):
        return -self._likelihood(x)


class RMSError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Calculates the square root of a normalised sum-of-squares error:
    ``f = sqrt( sum( (x[i] - y[i])**2 / n) )``
    """
    def __init__(self, problem):
        super(RMSError, self).__init__(problem)
        self._ninv = 1.0 / len(self._values)

    def __call__(self, x):
        return np.sqrt(self._ninv * np.sum(
            (self._problem.evaluate(x) - self._values)**2))


class SumOfSquaresError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Calculates a sum-of-squares error: ``f = sum( (x[i] - y[i])**2 )``
    """
    def __call__(self, x):
        return np.sum((self._problem.evaluate(x) - self._values)**2)

