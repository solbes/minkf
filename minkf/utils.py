import numpy as np


def rsolve(A, b):
    """
    Solve system x*A = b for x. Can be used to efficiently calculate x=b*inv(A).

    Parameters
    ----------
    A: np.array, system matrix
    b: np.array, right hand side

    Returns
    -------
    solution of x*A = b
    """
    return np.linalg.solve(A.T, b.T).T


def normal_log_pdf(x, mu, covmat):
    """
    Log-likelihood of a multivariate normal distribution.

    Parameters
    ----------
    x: np.array, input vector
    mu: np.array, mean vector
    covmat: np.array, covariance matrix

    Returns
    -------
    likelihood value as scalar
    """

    data_misfit = 0.5*np.sum((x-mu)*np.linalg.solve(covmat, x-mu))
    norm_constant = np.log(np.linalg.det(2 * np.pi * covmat))

    return data_misfit + norm_constant
