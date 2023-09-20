import pytensor
import scipy
from pytensor.scalar.basic import UnaryScalarOp
import numpy as np

def ndtr(mu, sigma, x):
    """
        Pytensor implementation of the normal distribution CDF

        Parameters
        ----------
        mu: float
            Mean of the normal distribution
        sigma: float
            Standard deviation of the normal distribution

        Returns
        -------
        float
            The CDF of the normal distribution at x
    """
    return 0.5*(1 + pytensor.tensor.erf((x-mu)/(sigma*np.sqrt(2))))