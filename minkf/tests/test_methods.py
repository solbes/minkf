import numpy as np
import minkf as kf


def test_filter_and_smoother():

    # case 1: 1d-signal, constant matrices
    y = np.ones(3)

    x0 = np.array([0.0])
    Cest0 = 1 * np.array([[1.0]])
    M = np.array([[1.0]])
    K = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])

    res = kf.run_filter(y, x0, Cest0, M, K, Q, R, likelihood=True)
    exp_x = [np.array([0.66666]), np.array([0.875]), np.array([0.952381])]
    exp_C = [np.array([[0.66666]]), np.array([[0.625]]), np.array([[0.619048]])]
    np.testing.assert_allclose(res['x'], exp_x, rtol=1e-4)
    np.testing.assert_allclose(res['C'], exp_C, rtol=1e-4)
    np.testing.assert_allclose(res['loglike'], 8.74862982742765)

    res_smo = kf.run_smoother(y, x0, Cest0, M, K, Q, R)
    exp_x_smo = [np.array([0.7619]), np.array([0.90476]), np.array([0.95238])]
    exp_C_smo = [np.array([[0.47619]]), np.array([[0.47619]]),
                 np.array([[0.61905]])]

    np.testing.assert_allclose(res_smo['x'], exp_x_smo, rtol=1e-4)
    np.testing.assert_allclose(res_smo['C'], exp_C_smo, rtol=1e-4)

    # case 2: 2d-signal, constant matrices
    y = [np.ones(2), np.ones(2), np.ones(2)]

    x0 = np.array([0.0, 0.0])
    Cest0 = np.eye(2)
    M = np.eye(2)
    K = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)

    res = kf.run_filter(y, x0, Cest0, M, K, Q, R, likelihood=True)
    exp_x = [np.array([0.66666667, 0.66666667]),
           np.array([0.875, 0.875]),
           np.array([0.95238095, 0.95238095])]
    exp_C = [np.array([[0.66666667, 0.],
                       [0., 0.66666667]]),
             np.array([[0.625, 0.],
                       [0., 0.625]]),
             np.array([[0.61904762, 0.],
                      [0., 0.61904762]])]

    np.testing.assert_allclose(res['x'], exp_x)
    np.testing.assert_allclose(res['C'], exp_C)
    np.testing.assert_allclose(res['loglike'], 17.4972596548553)

    res_smo = kf.run_smoother(y, x0, Cest0, M, K, Q, R)
    exp_x_smo = [np.array([0.76190476, 0.76190476]),
                 np.array([0.9047619, 0.9047619]),
                 np.array([0.95238095, 0.95238095])]
    exp_C_smo = [np.array([[0.47619048, 0.],
                          [0., 0.47619048]]),
                 np.array([[0.47619048, 0.],
                          [0., 0.47619048]]),
                 np.array([[0.61904762, 0.],
                          [0., 0.61904762]])]

    np.testing.assert_allclose(res_smo['x'], exp_x_smo)
    np.testing.assert_allclose(res_smo['C'], exp_C_smo)

    # case 3: 2d-signal, varying matrices, calculate y predictions
    y = [np.ones(2), np.ones(2), np.ones(2)]

    x0 = np.array([0.0, 0.0])
    Cest0 = np.eye(2)
    M = [np.eye(2), 2 * np.eye(2), 3 * np.eye(2)]
    K = [np.eye(2), 0.1 * np.eye(2), 0.01 * np.eye(2)]
    Q = [np.eye(2), 2 * np.eye(2), 4 * np.eye(2)]
    R = [np.eye(2), 0.5 * np.eye(2), 0.25 * np.eye(2)]

    res = kf.run_filter(
        y, x0, Cest0, M, K, Q, R,
        likelihood=True, predict_y=True
    )
    exp_x = [
        np.array([0.66666667, 0.66666667]),
        np.array([2.07317073, 2.07317073]),
        np.array([7.78403477, 7.78403477])
    ]
    exp_C = [
        np.array([
            [0.66666667, 0.        ],
            [0.        , 0.66666667]
        ]),
        np.array([
            [4.26829268, 0.        ],
            [0.        , 4.26829268]
        ]),
        np.array([
            [41.70703863,  0.        ],
            [ 0.        , 41.70703863]
        ])
    ]
    exp_yp = [
        np.array([0., 0.]),
        np.array([0.13333333, 0.13333333]),
        np.array([0.06219512, 0.06219512])
    ]
    exp_Cyp = [
        np.array([
            [3., 0.],
            [0., 3.]
        ]),
        np.array([
            [0.54666667, 0.        ],
            [0.        , 0.54666667]
        ]),
        np.array([
            [0.25424146, 0.        ],
            [0.        , 0.25424146]
        ])
    ]
    np.testing.assert_allclose(res['x'], exp_x)
    np.testing.assert_allclose(res['C'], exp_C)
    np.testing.assert_allclose(res['loglike'], 14.444253596157939)
    np.testing.assert_allclose(res["yp"], exp_yp)
    np.testing.assert_allclose(res["Cyp"], exp_Cyp)

    res_smo = kf.run_smoother(y, x0, Cest0, M, K, Q, R)
    exp_x_smo = [
        np.array([1.01299897, 1.01299897]),
        np.array([2.54549641, 2.54549641]),
        np.array([7.78403477, 7.78403477])
    ]
    exp_C_smo = [
        np.array([
            [0.6288817, 0.       ],
            [0.       , 0.6288817]]),
        np.array([
            [4.20380088, 0.        ],
            [0.        , 4.20380088]
        ]),
        np.array([
            [41.70703863,  0.        ],
            [ 0.        , 41.70703863]
        ])
    ]
    np.testing.assert_allclose(res_smo['x'], exp_x_smo)
    np.testing.assert_allclose(res_smo['C'], exp_C_smo)

    return
