import numpy as np


def kf_predict(xest, Cest, M, Q):
    return M.dot(xest), M.dot(Cest.dot(M.T)) + Q


def kf_update(y, xp, Cp, K, R):

    CpKT = Cp.dot(K.T)
    obs_precision = K.dot(CpKT) + R
    xest = xp + CpKT.dot(np.linalg.solve(obs_precision, y-K.dot(xp)))
    Cest = Cp - CpKT.dot(np.linalg.solve(obs_precision), CpKT.T)

    return xest, Cest


def run_filter(y, x0, Cest0, M, K, Q, R):

    # TODO: ensure everything is a list of np.arrays

    xest = x0
    Cest = Cest0
    nobs = len(y)

    xp_all = nobs*[None]
    Cp_all = nobs*[None]
    xest_all = nobs*[None]
    Cest_all = nobs*[None]

    for i in range(nobs):

        xp, Cp = kf_predict(xest, Cest, M[i], Q[i])
        xest, Cest = kf_update(y[i], xp, Cp, K[i], R[i])

        xp_all[i] = xp
        Cp_all[i] = Cp
        xest_all[i] = xest
        Cest_all[i] = Cest

    results = {
        'xp': xp_all,
        'Cp': Cp_all,
        'xest': xest_all,
        'Cest': Cest_all,
    }

    return results
