from random import randint, seed

import numpy as np

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]

EPSILON = 1e-8


def almost_equal(a, b):
    return abs(a - b) <= EPSILON


class Classifier:
    def __init__(self):
        pass

    def classify(self, point):
        raise NotImplementedError

    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        :param data: A list of datapoints
        :param labels: The list of labels we correlate against (+1 / -1)
        """

        m = len(data)
        assert m == len(labels), "Data and labels must be the same size %i vs %i" % (m, len(labels))
        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

        predicts = [1 if self.classify(point) else -1 for point in data]
        return float(np.dot(labels, predicts)) / m


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        :param x: First dimension
        :param y: Second dimension
        :param b: Bias term
        """
        Classifier.__init__(self)
        self._vector = np.asarray([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) - 0 >= -EPSILON

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """

    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        :param x: First dimension
        :param y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.
    """

    def __init__(self, start_x, start_y, end_x, end_y):
        """
        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        :param start_x: Left position
        :param start_y: Bottom position
        :param end_x: Right position
        :param end_y: Top position
        """
        Classifier.__init__(self)
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
                                 (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
                                 (end_y, start_y)
        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        :param point: The point to classify
        """
        return (self._x1 <= point[0] <= self._x2) and (self._y1 <= point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    :param dataset: The dataset to use to generate hypotheses
    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    :param dataset: The dataset to use to generate hypotheses
    """
    inter_vector = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])

    if len(dataset) == 1 and almost_equal(dataset[0][0], 0) and almost_equal(dataset[0][1], 0):
        yield OriginPlaneHypothesis(1., 1.)
    else:
        norm_vectors = np.asarray(
            [np.asarray((v[0] if v[1] >= 0 else -v[0], v[1]) / np.linalg.norm(v)) for v in dataset])
        norm_vectors = norm_vectors[norm_vectors[:, 0].argsort()][::-1]

        uni_list, oppo_list = [], []
        cv = None
        for nv in norm_vectors:
            if cv is not None and almost_equal(nv[0], cv[0]):
                if not almost_equal(nv[1], cv[1]) and (not oppo_list or not almost_equal(oppo_list[-1][0], nv[0])):
                    oppo_list.append((nv[0], abs(nv[1])))
            else:
                cv = nv
                uni_list.append((nv[0], abs(nv[1])))
        if almost_equal(uni_list[0][0], 1.) and almost_equal(uni_list[-1][0], -1.):
            oppo_list.append((1., 0.))
            uni_list.pop()

        if len(uni_list) == 1:
            if not almost_equal(uni_list[0][0], 0):
                yield OriginPlaneHypothesis(1., 0)
                yield OriginPlaneHypothesis(-1., 0)
            else:
                yield OriginPlaneHypothesis(0, 1.)
                yield OriginPlaneHypothesis(0, -1.)
        else:
            tl = [(-uni_list[-1][0], -uni_list[-1][1])] + uni_list + [(-uni_list[0][0], -uni_list[0][1])]
            for i in range(len(tl) - 1):
                if i == 0 or i == len(tl) - 2:
                    v = inter_vector(tl[i], tl[i + 1])
                    yield OriginPlaneHypothesis(-v[1], v[0])
                else:
                    v = inter_vector(tl[i], tl[i + 1])
                    yield OriginPlaneHypothesis(-v[1], v[0])
                    yield OriginPlaneHypothesis(v[1], -v[0])
        for v in oppo_list:
            yield OriginPlaneHypothesis(-v[1], v[0])
            if len(uni_list) != 1:
                yield OriginPlaneHypothesis(v[1], -v[0])


def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    :param dataset: The dataset to use to generate hypotheses
    """

    # TODO: Complete this for extra credit
    yield PlaneHypothesis(1.0, 0.0, 0.0)


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    :param dataset: The dataset to use to generate hypotheses
    """

    # TODO: complete this function
    yield AxisAlignedRectangle(0, 0, 0, 0)


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    :param number: The number of coin tosses to perform
    :param random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500, random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    :param dataset: a sequence of examples that can be handled by the hypotheses generated by the hypothesis_generator
    :param hypothesis_generator: a function that generates an iterator over hypotheses given a dataset
    :param num_samples: the number of samples to use in estimating the Rademacher correlation
    :param random_seed: The random seed to use
    """

    # TODO: complete this function
    return 0.0


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
