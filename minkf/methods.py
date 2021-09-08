import numpy as np


def kf_predict(xest, Cest, M, Q, u=None):

    xp = M.dot(xest) if u is None else M.dot(xest) + u
    Cp = M.dot(Cest.dot(M.T)) + Q

    return xp, Cp


def kf_update(y, xp, Cp, K, R):

    CpKT = Cp.dot(K.T)
    obs_precision = K.dot(CpKT) + R
    G = np.linalg.solve(obs_precision.T, CpKT.T).T

    xest = xp + G.dot(y-K.dot(xp))
    Cest = Cp - G.dot(CpKT.T)

    return xest, Cest


def run_filter(y, x0, Cest0, M, K, Q, R, u=None):

    xest = x0
    Cest = Cest0
    nobs = len(y)

    # transform inputs into lists of np.arrays (unless they already are)
    Mlist = M if type(M) == list else nobs * [M]
    Klist = K if type(M) == list else nobs * [K]
    Qlist = Q if type(M) == list else nobs * [Q]
    Rlist = R if type(M) == list else nobs * [R]
    ulist = u if type(u) == list else nobs * [u]

    # space for saving end results
    xp_all = nobs * [None]
    Cp_all = nobs * [None]
    xest_all = nobs * [None]
    Cest_all = nobs * [None]

    for i in range(nobs):

        xp, Cp = kf_predict(xest, Cest, Mlist[i], Qlist[i], u=ulist[i])
        xest, Cest = kf_update(y[i], xp, Cp, Klist[i], Rlist[i])

        xp_all[i] = xp
        Cp_all[i] = Cp
        xest_all[i] = xest
        Cest_all[i] = Cest

    results = {
        'xp': xp_all,
        'Cp': Cp_all,
        'x': xest_all,
        'C': Cest_all,
    }

    return results
