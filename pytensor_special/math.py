from .elemwise import scalar_elemwise
import pytensor
import numpy

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
    return 0.5*(1 + pytensor.tensor.erf((x-mu)/(sigma*numpy.sqrt(2))))

@scalar_elemwise
def gammaincinv(a, z):
    """
        Pytensor implementation of the inverse incomplete gamma function
    """

@scalar_elemwise
def betaincinv(a, b, z):
    """
        Pytensor implementation of the inverse incomplete beta function
    """