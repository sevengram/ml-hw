from random import randint, seed

import numpy

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


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

        predicts = [self.classify(point) for point in data]
        return float(numpy.dot(labels, predicts)) / m


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
        self._vector = numpy.array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return 1 if self(point) >= 0 else -1

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
        return 1 if (self._x1 <= point[0] <= self._x2) and (self._y1 <= point[1] <= self._y2) else -1

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return 1


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

    # TODO: Complete this function
    yield OriginPlaneHypothesis(1.0, 0.0)


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
