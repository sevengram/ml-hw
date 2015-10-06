import numpy as np

kINSP = np.array([(1, 8, +1),
                  (7, 2, -1),
                  (6, -1, -1),
                  (-5, 0, +1),
                  (-5, 1, -1),
                  (-5, 2, +1),
                  (6, 3, +1),
                  (6, 1, -1),
                  (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),  # 0 - A
                 (0, 4, +1),  # 1 - B
                 (2, 1, +1),  # 2 - C
                 (-2, -3, -1),  # 3 - D
                 (0, -1, -1),  # 4 - E
                 (2, -3, -1),  # 5 - F
                 ])


def almost_eq(a, b, tolerance):
    return abs(a - b) < tolerance


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """
    w = np.zeros(len(x[0]))
    for i in range(len(x)):
        w += alpha[i] * y[i] * x[i]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """
    return set(i for i in range(len(x)) if almost_eq(y[i] * (np.dot(w, x[i]) + b), 1, tolerance))


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """
    return set(i for i in range(len(x)) if y[i] * (np.dot(w, x[i]) + b) < 0)
