import argparse
from collections import defaultdict

import numpy as np
from sklearn import svm

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


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        import cPickle
        import gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


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


def confusion_matrix(classifier, test_x, test_y):
    """
    Given a matrix of test examples and labels, compute the confusion
    matrix for the current classifier. Should return a dictionary of
    dictionaries where d[ii][jj] is the number of times an example
    with true label ii was labeled as jj.

    :param test_x: Test data representation
    :param test_y: Test data answers
    """
    d = defaultdict(dict)
    data_index = 0
    for example, answer in zip(test_x, test_y):
        label = classifier.predict(example)[0]
        d[answer][label] = d[answer].get(label, 0) + 1
        data_index += 1
        if data_index % 100 == 0:
            print("%i/%i for confusion matrix" % (data_index, len(test_x)))
    return d


def accuracy(conf_matrix):
    """
    Given a confusion matrix, compute the accuracy of the underlying classifier.
    """

    total = 0
    correct = 0
    for i in conf_matrix:
        total += sum(conf_matrix[i].values())
        correct += conf_matrix[i].get(i, 0)

    if total:
        return float(correct) / float(total)
    else:
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--kernel', type=str, default='rbf',
                        help="Specifies the kernel type to be used in the algorithm. It must be one of "
                             "'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'. Default is 'rbf'.")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    train_set_x, train_set_y = [], []
    for ii in range(len(data.train_y)):
        if data.train_y[ii] in (3, 8):
            train_set_x.append(data.train_x[ii])
            train_set_y.append(data.train_y[ii])
    test_set_x, test_set_y = [], []
    for ii in range(len(data.test_y)):
        if data.test_y[ii] in (3, 8):
            test_set_x.append(data.test_x[ii])
            test_set_y.append(data.test_y[ii])
    print("Done loading data")

    clf = svm.SVC(kernel=args.kernel)
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        clf.fit(train_set_x[:args.limit], train_set_y[:args.limit])
    else:
        clf.fit(train_set_x, train_set_y)

    confusion = confusion_matrix(clf, test_set_x, test_set_y)
    print("\t" + "\t".join(str(x) for x in [3, 8]))
    print("".join(["-"] * 30))
    for ii in [3, 8]:
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0)) for x in [3, 8]))
    print("Accuracy: %f" % accuracy(confusion))
