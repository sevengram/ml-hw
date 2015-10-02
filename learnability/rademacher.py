import bisect
from random import randint, seed
import time

import numpy as np

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


def almost_eq(a, b):
    return abs(a - b) <= 1e-8


def almost_eq_point(a, b):
    return a and b and almost_eq(a[0], b[0]) and almost_eq(a[1], b[1])


def almost_eq_x(a, b):
    return a and b and almost_eq(a[0], b[0])


def almost_eq_y(a, b):
    return a and b and almost_eq(a[1], b[1])


def almost_geq(a, b):
    return a - b >= -1e-8


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

    def vector(self):
        return self._vector

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return almost_geq(self(point), 0.0)

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


def _is_original_point(p):
    return almost_eq(p[0], 0) and almost_eq(p[1], 0)


def _all_original_point(data):
    for d in data:
        if not _is_original_point(d):
            return False
    return True


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    :param dataset: The dataset to use to generate hypotheses
    """
    inter_vector = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])

    if _all_original_point(dataset):
        yield OriginPlaneHypothesis(1., 1.)
    else:
        norm_vectors = np.asarray(
            [np.asarray((v[0] if almost_geq(v[1], 0) else -v[0], v[1]) / np.linalg.norm(v)) for v in dataset if
             not _is_original_point(v)])
        norm_vectors = norm_vectors[norm_vectors[:, 0].argsort()][::-1]

        uni_list, oppo_list = [], []
        cv = None
        for nv in norm_vectors:
            if cv is not None and almost_eq(nv[0], cv[0]):
                if not almost_eq(nv[1], cv[1]) and (not oppo_list or not almost_eq(oppo_list[-1][0], nv[0])):
                    oppo_list.append((nv[0], abs(nv[1])))
            else:
                cv = nv
                uni_list.append((nv[0], abs(nv[1])))
        if almost_eq(uni_list[0][0], 1.) and almost_eq(uni_list[-1][0], -1.):
            oppo_list.append((1., 0.))
            uni_list.pop()

        if len(uni_list) == 1:
            if not almost_eq(uni_list[0][0], 0):
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


def _classify_res_bits(h, dataset):
    res = 0
    for i in range(len(dataset)):
        if h.classify(dataset[i]):
            res |= (1 << i)
    return res


def _point_transform(old, new_origin):
    return np.asarray(old) - np.asarray(new_origin)


def _get_bias(vector, new_origin):
    return np.dot(vector, new_origin)


def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    :param dataset: The dataset to use to generate hypotheses
    """
    unique_res = set()
    yield PlaneHypothesis(1, 0, max(dataset)[0] + 1)
    for new_origin in dataset:
        new_points = [_point_transform(p, new_origin) for p in dataset]
        for oph in origin_plane_hypotheses(new_points):
            res = _classify_res_bits(oph, new_points)
            if res not in unique_res:
                unique_res.add(res)
                v = oph.vector()
                yield PlaneHypothesis(v[0], v[1], _get_bias(v, new_origin))


def _get_a_case(a0, a1, b):
    if a0 is None or a0[0] != a1[0]:
        return 0
    if b[1] > a0[1]:
        return 1
    else:
        return 2


def _get_b_case(a, b0, b1, ys):
    if b0 is None or b0[0] != b1[0]:
        return 0
    if a[1] >= b1[1]:
        return 1
    if b0[1] < a[1] < b1[1]:
        if b1[1] not in ys:
            return 2
        if b1[1] in ys:
            return 3
    if a[1] <= b0[1]:
        if b1[1] not in ys:
            return 4
        if b1[1] in ys:
            return 5


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    :param dataset: The dataset to use to generate hypotheses
    """
    dataset.sort()
    yield AxisAlignedRectangle(dataset[0][0] - 1, 0, dataset[0][0] - 1, 0)
    m = len(dataset)
    mid_pt_map, mid_pt_ys = {}, []
    last_a = None
    for i in range(m):
        a = dataset[i]
        if almost_eq_point(a, last_a):
            continue
        yield AxisAlignedRectangle(a[0], a[1], a[0], a[1])
        mid_pt_map.clear()
        del mid_pt_ys[:]
        last_b = None
        left = a[0]
        for j in range(i + 1, m):
            b = dataset[j]
            if almost_eq_point(b, a) or almost_eq_point(b, last_b):
                continue
            right, top, bottom = b[0], max(a[1], b[1]), min(a[1], b[1])
            a_case = _get_a_case(last_a, a, b)
            b_case = _get_b_case(a, last_b, b, mid_pt_map)
            if a_case != 2 and b_case != 5:
                yield AxisAlignedRectangle(left, bottom, right, top)

            ay = last_a[1] if a_case == 1 else None
            bi = bisect.bisect_right(mid_pt_ys, last_b[1]) if last_b else 0
            lo = bisect.bisect_right(mid_pt_ys, top)
            hi = bisect.bisect_left(mid_pt_ys, bottom)
            high_points = [mid_pt_map[i] for i in mid_pt_ys[lo:]]
            if a_case != 2:
                if b_case <= 4:
                    li = 0 if b_case % 2 == 0 else bi
                    for q in [mid_pt_map[i] for i in mid_pt_ys[li:hi]]:
                        if q[1] > ay:
                            yield AxisAlignedRectangle(left, q[1], right, top)
                if b_case <= 3:
                    li = 0 if b_case == 0 else bi
                    for p in high_points:
                        yield AxisAlignedRectangle(left, bottom, right, p[1])
                        for q in [mid_pt_map[i] for i in mid_pt_ys[li:hi]]:
                            if q[1] > ay:
                                yield AxisAlignedRectangle(left, q[1], right, p[1])
            if b[1] not in mid_pt_map:
                mid_pt_map[b[1]] = b
                bisect.insort(mid_pt_ys, b[1])
            last_b = b
        last_a = a


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

    hyp_set = list(hypothesis_generator(dataset))
    m = len(dataset)
    max_correlations = []
    for i in range(num_samples):
        sigma = coin_tosses(m, random_seed if i == 0 else 0)
        max_correlations.append(max(hyp.correlation(dataset, sigma) for hyp in hyp_set))
    return np.mean(max_correlations)


############
# Brute force method for hypotheses, kept for double check the result

def origin_plane_hypotheses2(dataset):
    if _all_original_point(dataset):
        yield OriginPlaneHypothesis(1., 1.)
    else:
        inter_vector = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])
        norm_vectors = np.asarray(
            [np.asarray((v[0] if almost_geq(v[1], 0) else -v[0], abs(v[1])) / np.linalg.norm(v)) for v in dataset if
             not _is_original_point(v)])
        norm_vectors = norm_vectors[norm_vectors[:, 0].argsort()][::-1]

        inter_points = []
        uniq_result = set()
        for i in range(len(norm_vectors)):
            iv = inter_vector(norm_vectors[i], norm_vectors[i - 1])
            if almost_eq_point(iv, (0, 0)):
                inter_points += [(1, 0), (0, 1)]
            else:
                inter_points.append(iv)
        iv = inter_vector(norm_vectors[0], -np.asarray(norm_vectors[-1]))
        if almost_eq_point(iv, (0, 0)):
            inter_points += [(1, 0), (0, 1)]
        else:
            inter_points.append(iv)
        for p in list(norm_vectors) + inter_points:
            for h in [OriginPlaneHypothesis(-p[1], p[0]), OriginPlaneHypothesis(p[1], -p[0])]:
                res = _classify_res_bits(h, dataset)
                if res not in uniq_result:
                    uniq_result.add(res)
                    yield h


def _bit_assign(indices):
    res = 0
    for i in indices:
        res |= (1 << i)
    return res


def _search_combines_recur(items, low, high, depth, start, result, seq):
    for i in range(len(items) - 1, start - 1, -1):
        seq[depth] = items[i]
        if depth + 1 >= low:
            result.append([n for n in seq if n is not None])
        if depth + 1 < high:
            _search_combines_recur(items, low, high, depth + 1, i + 1, result, seq)
        seq[depth] = None


def _range_combines(items, low, high):
    if not items:
        return []
    low = max(low, 1)
    high = max(min(high, len(items)), low)
    result = []
    _search_combines_recur(items, low, high, 0, 0, result, [None] * high)
    return result


def _is_boundary_point(point, left, right, low, high):
    x0 = point[0]
    x1 = point[1]
    return ((x0 == left or x0 == right) and low <= x1 <= high) or ((x1 == low or x1 == high) and left <= x0 <= right)


def _generate_rectangle(points, rect_indices):
    l = len(rect_indices)
    if l == 0:
        return [], []
    rect_points = list(np.asarray(points)[rect_indices])
    left = rect_points[0][0]
    right = rect_points[l - 1][0]
    high = max(rect_points, key=lambda x: x[1])[1]
    low = min(rect_points, key=lambda x: x[1])[1]

    # scan the boundary points and coverd points
    boundary_indices, cover_indices = [], []
    i = rect_indices[0]
    while i >= 0 and _is_boundary_point(points[i], left, right, low, high):
        boundary_indices.append(i)
        i -= 1
    i = rect_indices[l - 1]
    while i < len(points) and _is_boundary_point(points[i], left, right, low, high):
        boundary_indices.append(i)
        i += 1
    for i in range(rect_indices[0] + 1, rect_indices[l - 1]):
        if _is_boundary_point(points[i], left, right, low, high):
            boundary_indices.append(i)
        if l < 4 and i not in rect_indices and low <= points[i][1] <= high:
            cover_indices.append(i)
    return (left, low, right, high), boundary_indices, _range_combines(cover_indices, 1, 4 - l)


def axis_aligned_hypotheses2(dataset):
    dataset.sort()
    yield AxisAlignedRectangle(dataset[0][0] - 1, 0, dataset[0][0] - 1, 0)
    scaned_pairs, boundary_pairs = set(), set()
    for rect_indices in _range_combines(range(len(dataset)), 1, 4):
        n = _bit_assign(rect_indices)
        if n not in scaned_pairs:
            scaned_pairs.add(n)
            boundary, boundary_indices, covered_pairs = _generate_rectangle(dataset, rect_indices)
            b = _bit_assign(boundary_indices)
            if b not in boundary_pairs:
                boundary_pairs.add(b)
                yield AxisAlignedRectangle(*boundary)
            for covered_indices in covered_pairs:
                scaned_pairs.add(_bit_assign(rect_indices + covered_indices))


# End of brute force method
############

if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses, num_samples=1000, random_seed=int(time.time())))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses, num_samples=1000,
                              random_seed=int(time.time())))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses, num_samples=1000,
                              random_seed=int(time.time())))
