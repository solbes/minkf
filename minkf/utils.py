import numpy as np


def normal_log_pdf(x, mu, covmat):

    data_misfit = 0.5*np.sum((x-mu)*np.linalg.solve(covmat, x-mu))
    norm_constant = np.log(np.linalg.det(2 * np.pi * covmat))

    return data_misfit + norm_constant
