import numpy as np
from minkf import utils


def kf_predict(xest, Cest, M, Q, u=None):
    """
    Prediction step of the Kalman filter, xp = M*xest + u + Q.

    Parameters
    ----------
    xest: np.array of length n, posterior estimate for the current time
    Cest: np.array of shape (n, n), posterior covariance for the current step
    M: np.array of shape (n, n), dynamics model matrix
    Q: np.array of shape (n, n), model error covariance matrix
    u: np.array of length n, optional control input

    Returns
    -------
    xp, Cp: predicted mean and covariance
    """

    xp = M.dot(xest) if u is None else M.dot(xest) + u
    Cp = M.dot(Cest.dot(M.T)) + Q

    return xp, Cp


def kf_update(y, xp, Cp, K, R):
    """
    Update step of the Kalman filter

    Parameters
    ----------
    y: np.array of length m, observation vector
    xp: np.array of length n, predicted (prior) mean
    Cp: np.array of shape (n, n), predicted (prior) covariance
    K: np.array of shape (m, n), observation model matrix
    R: np.array of shape (m, m), observation error covariance

    Returns
    -------
    xest, Cest: estimated mean and covariance
    """

    CpKT = Cp.dot(K.T)
    obs_precision = K.dot(CpKT) + R
    G = utils.rsolve(obs_precision, CpKT)

    xest = xp + G.dot(y-K.dot(xp))
    Cest = Cp - G.dot(CpKT.T)

    return xest, Cest


def run_filter(
        y, x0, Cest0, M, K, Q, R,
        u=None, likelihood=False, predict_y=False
):
    """
    Run the Kalman filter.

    Parameters
    ----------
    y: list of np.arrays of length m, observation vectors
    x0: np.array of length n, initial state
    Cest0: np.array of shape (n, n), initial covariance
    M: np.array or list of np.arrays of shape (n, n), dynamics model matrices
    K: np.array or list of np.arrays of shape (m, n), observation model matrices
    Q: np.array or list of np.arrays of shape (n, n), model error covariances
    R: np.array or list of np.arrays of shape (m, m), obs error covariances
    u: list of np.arrays of length n, optional control inputs
    likelihood: bool, calculate likelihood along with the filtering
    predict_y: bool, include predicted observations and covariance

    Returns
    -------
    results dict = {
        'x': estimated states
        'C': covariances for the estimated states
        'loglike': log-likelihood of the data if calculated, None otherwise
        'xp': predicted means (helper variables for further calculations)
        'Cp': predicted covariances (helpers for further calculations)
        'yp': observation predicted at the predicted mean state
        'Cyp': covariance of predicted observation
    }
    """

    xest = x0
    Cest = Cest0
    nobs = len(y)

    # transform inputs into lists of np.arrays (unless they already are)
    Mlist = M if type(M) == list else nobs * [M]
    Klist = K if type(K) == list else nobs * [K]
    Qlist = Q if type(Q) == list else nobs * [Q]
    Rlist = R if type(R) == list else nobs * [R]
    uList = u if type(u) == list else nobs * [u]

    # space for saving end results
    xp_all = nobs * [None]
    Cp_all = nobs * [None]
    xest_all = nobs * [None]
    Cest_all = nobs * [None]

    loglike = 0 if likelihood else None
    yp_all = nobs * [None] if predict_y else None
    Cyp_all = nobs * [None] if predict_y else None
    for i in range(nobs):

        Ki = Klist[i]
        Ri = Rlist[i]
        xp, Cp = kf_predict(xest, Cest, Mlist[i], Qlist[i], u=uList[i])
        xest, Cest = kf_update(y[i], xp, Cp, Ki, Ri)

        xp_all[i] = xp
        Cp_all[i] = Cp
        xest_all[i] = xest
        Cest_all[i] = Cest

        if likelihood:
            yp = Ki.dot(xp)
            Cyp = Ki.dot(Cp).dot(Ki.T) + Ri
            loglike += utils.normal_log_pdf(y[i], yp, Cyp)
            if predict_y:
                yp_all[i] = yp
                Cyp_all[i] = Cyp


    results = {
        'xp': xp_all,
        'Cp': Cp_all,
        'yp': yp_all,
        'Cyp': Cyp_all,
        'x': xest_all,
        'C': Cest_all,
        'loglike': loglike
    }

    return results


def run_smoother(y, x0, Cest0, M, K, Q, R, u=None):
    """
    Run the Kalman smoother.

    Parameters
    ----------
    y: list of np.arrays of length m, observation vectors
    x0: np.array of length n, initial state
    Cest0: np.array of shape (n, n), initial covariance
    M: np.array or list of np.arrays of shape (n, n), dynamics model matrices
    K: np.array or list of np.arrays of shape (m, n), observation model matrices
    Q: np.array or list of np.arrays of shape (n, n), model error covariances
    R: np.array or list of np.arrays of shape (m, m), obs error covariances
    u: list of np.arrays of length n, optional control inputs

    Returns
    -------
    results dict = {
        'x': smoothed states
        'C': covariances for the smoothed states
    }
    """

    # run the filter first
    kf_res = run_filter(y, x0, Cest0, M, K, Q, R, u=u)
    xest = kf_res['x']
    Cest = kf_res['C']
    xp = kf_res['xp']
    Cp = kf_res['Cp']

    nobs = len(y)
    xs = nobs * [None]
    Cs = nobs * [None]
    xs[-1] = xest[-1]
    Cs[-1] = Cest[-1]

    # transform inputs into lists of np.arrays (unless they already are)
    Mlist = M if type(M) == list else nobs * [M]

    # backward recursion
    for i in range(nobs-2, -1, -1):
        G = utils.rsolve(Cp[i+1], Cest[i].dot(Mlist[i+1].T))
        xs[i] = xest[i] + G.dot(xs[i+1] - xp[i+1])
        Cs[i] = Cest[i] + G.dot(Cs[i+1] - Cp[i+1]).dot(G.T)

    results = {
        'x': xs,
        'C': Cs
    }

    return results


def sample(res_kf, M, Q, u=None, nsamples=1):
    """
    Sample from the posterior of the states given the observations.

    Parameters
    ----------
    res_kf: results dict from the Kalman filter run
    M: np.array or list of np.arrays of shape (n, n), dynamics model matrices
    Q: np.array or list of np.arrays of shape (n, n), model error covariances
    u: list of np.arrays of length n, optional control inputs
    nsamples: int, number of samples generated

    Returns
    -------
    list of state samples
    """

    nobs = len(res_kf['x'])
    ns = len(res_kf['x'][0])

    Mlist = M if type(M) == list else nobs * [M]
    Qlist = Q if type(M) == list else nobs * [Q]
    uList = u if u is not None else nobs * [np.zeros(ns)]

    MP_k = [M.dot(P) for M, P in zip(Mlist, res_kf['C'])]
    MtQinv = [utils.rsolve(Q, M.T) for M, Q in zip(Mlist, Qlist)]
    Sig_k = [P - MP.T.dot(np.linalg.solve(MP.dot(M.T) + Q, MP))
             for P, MP, Q in zip(res_kf['C'], MP_k, Qlist)]

    def sample_one():

        xsample = np.zeros((nobs, ns))
        xsample[-1] = np.random.multivariate_normal(
            res_kf['x'][-1], res_kf['C'][-1]
        )

        for i in reversed(range(nobs-1)):
            mu_i = Sig_k[i].dot(
                MtQinv[i].dot(
                    xsample[i+1, :]-uList[i]
                ) + np.linalg.solve(res_kf['C'][i], res_kf['x'][i])
            )
            xsample[i] = np.random.multivariate_normal(mu_i, Sig_k[i])

        return np.squeeze(xsample)

    return [sample_one() for i in range(nsamples)]
