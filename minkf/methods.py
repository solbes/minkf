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


def run_filter(y, x0, Cest0, M, K, Q, R, u=None, likelihood=False):
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

    Returns
    -------
    results dict = {
        'x': estimated states
        'C': covariances for the estimated states
        'loglike': log-likelihood of the data if calculated, None otherwise
        'xp': predicted means (helper variables for further calculations)
        'Cp': predicted covariances (helpers for further calculations)
    }
    """

    xest = x0
    Cest = Cest0
    nobs = len(y)

    # transform inputs into lists of np.arrays (unless they already are)
    Mlist = M if type(M) == list else nobs * [M]
    Klist = K if type(M) == list else nobs * [K]
    Qlist = Q if type(M) == list else nobs * [Q]
    Rlist = R if type(M) == list else nobs * [R]

    # space for saving end results
    xp_all = nobs * [None]
    Cp_all = nobs * [None]
    xest_all = nobs * [None]
    Cest_all = nobs * [None]

    loglike = 0 if likelihood else None
    for i in range(nobs):

        xp, Cp = kf_predict(xest, Cest, Mlist[i], Qlist[i], u=u[i])
        xest, Cest = kf_update(y[i], xp, Cp, Klist[i], Rlist[i])

        xp_all[i] = xp
        Cp_all[i] = Cp
        xest_all[i] = xest
        Cest_all[i] = Cest

        if likelihood:
            yp = K.dot(xp)
            Cyp = K.dot(Cp).dot(K.T) + R
            loglike += utils.normal_log_pdf(y[i], yp, Cyp)

    results = {
        'xp': xp_all,
        'Cp': Cp_all,
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

    # backward recursion
    for i in range(nobs-2, -1, -1):
        G = utils.rsolve(Cp[i+1], Cest[i].dot(M.T))
        xs[i] = xest[i] + G.dot(xs[i+1] - xp[i+1])
        Cs[i] = Cest[i] + G.dot(Cs[i+1] - Cp[i+1]).dot(G.T)

    results = {
        'x': xs,
        'C': Cs
    }

    return results
