import unittest

from rademacher import origin_plane_hypotheses, rademacher_estimate, PlaneHypothesis, \
    constant_hypotheses, axis_aligned_hypotheses
from vc_sin import train_sin_classifier

rand_data1 = [(-4, -8), (-4, -5), (1, 7), (0, 6), (-9, 0), (0, -2), (2, -4), (0, -3), (-7, 7), (-7, 5), (4, 1),
              (-8, -9), (10, 10), (-3, 4), (4, 2), (-1, 4), (-5, -6), (-9, -4), (9, -9), (8, 8), (-6, -10), (-5, -4),
              (6, -8), (4, 0), (-6, 4), (-5, 7), (-9, 6), (-4, -6), (0, 8), (-6, -5), (-5, -6), (-10, 1), (1, 4),
              (2, -7), (8, 7), (-3, 6), (-2, -9), (1, -2), (9, 4), (-6, 1), (9, 0), (8, -5), (-8, -2), (10, -8),
              (-9, -9), (-5, 9), (-4, 1), (1, -3), (-1, 2), (-5, 0), (1, 0), (7, -1), (1, 9), (9, -9), (-7, -2),
              (-10, -9), (-2, 10), (-5, 5), (-5, 10), (-5, -2), (0, 7), (-4, 10), (-5, -5), (-3, 5), (3, -3),
              (-6, 10), (-5, -5), (4, 8), (0, -8), (-2, -9), (-9, 9), (-7, -10), (2, -6), (-5, -8), (8, 8), (-8, -1),
              (-3, 3), (3, 10), (3, -10), (-10, -4), (-9, 4), (2, -2), (-1, 8), (-6, -5), (-2, -10), (-4, 9), (7, 7),
              (-4, 3), (7, -1), (-4, 9), (-2, 5), (9, 1), (-1, 2), (7, 5), (-9, 1), (8, 6), (1, 4), (-10, -3),
              (5, -3), (10, 2)]

rand_data2 = [(-10, -1), (25, -17), (47, -45), (-25, 27), (-27, 9), (-44, -46), (6, 39), (-34, -44), (-37, -28),
              (-11, 14), (17, 30), (22, -6), (18, -32), (-17, -38), (-16, -31), (-37, 18), (-39, -22), (11, 31),
              (43, 34), (-16, 8)]


def assign_exists(data, classifiers, pattern):
    """
    Given a dataset and set of classifiers, make sure that the classification
    pattern specified exists somewhere in the classifier set.
    """

    val = False
    assert len(data) == len(pattern), "Length mismatch between %s and %s" % \
                                      (str(data), str(pattern))
    for hh in classifiers:
        present = all(hh.classify(data[x]) == pattern[x] for
                      x in xrange(len(data)))
        if present:
            print("%s matches %s" % (str(hh), str(pattern)))
        val = val or present
    if not val:
        print("%s not found in:" % str(pattern))
        for hh in classifiers:
            print("\t%s %s" % (str(hh), [hh.classify(x) for x in data]))
    return val


rect_hypo = axis_aligned_hypotheses


class TestLearnability(unittest.TestCase):
    def setUp(self):
        self._2d = {
            1: [(3, 3)],
            2: [(3, 3), (3, 4)],
            3: [(3, 3), (3, 4), (4, 3)],
            4: [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]}

        self._hypotheses = lambda x: [PlaneHypothesis(0, 0, 5),
                                      PlaneHypothesis(0, 0, -5),
                                      PlaneHypothesis(0, 1, 0),
                                      PlaneHypothesis(0, -1, 0),
                                      PlaneHypothesis(1, 0, 0),
                                      PlaneHypothesis(-1, 0, 0)]
        self._full_shatter = [(1, 1), (-1, -1)]
        self._half_shatter = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def test_rec_random(self):
        # data = []
        #     for i in range(20):
        #         data.append((0, randint(0, 40)))
        #         data.append((randint(0, 10), randint(-30, 70)))
        #         data.append((10, randint(0, 40)))
        #     count1 = count2 = 0
        #     for i in axis_aligned_hypotheses(data):
        #         count1 += 1
        #     print(count1)
        #     for i in axis_aligned_hypotheses2(data):
        #         count2 += 1
        #     print(count2)
        #     self.assertEqual(count2,count1)
        #     data = []
        #     for i in range(50):
        #         data.append((randint(-40, 40), randint(-40, 40)))
        #     count1 = count2 = 0
        #     for i in axis_aligned_hypotheses(data):
        #         count1 += 1
        #     print(count1)
        #     for i in axis_aligned_hypotheses2(data):
        #         count2 += 1
        #     print(count2)
        #     self.assertEqual(count2,count1)
        count1 = 0
        for i in axis_aligned_hypotheses(rand_data2):
            count1 += 1
        self.assertEqual(count1, 1340)

    def test_rec_single_point(self):
        hyps = list(rect_hypo(self._2d[1]))
        self.assertTrue(assign_exists(self._2d[1], hyps, [True]))
        self.assertTrue(assign_exists(self._2d[1], hyps, [False]))

    def test_rec_two_points(self):
        hyps = list(rect_hypo(self._2d[2]))
        for pp in [[False, False], [False, True], [True, False], [True, True]]:
            self.assertTrue(assign_exists(self._2d[2], hyps, pp))

    def test_rec_three_points(self):
        hyps = list(rect_hypo(self._2d[3]))
        for pp in [[False, False, False],
                   [False, True, False],
                   [True, False, False],
                   [False, False, True],
                   [True, True, False],
                   [True, False, True],
                   [True, True, True]]:
            self.assertTrue(assign_exists(self._2d[3], hyps, pp))
        data = [(0, 1), (0, 2), (1, -1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(7, len(hyps))
        data = [(0, 1), (0, 2), (1, 1.5)]
        hyps = list(rect_hypo(data))
        self.assertEqual(8, len(hyps))
        data = [(0, 1), (0, 2), (1, 5.6)]
        hyps = list(rect_hypo(data))
        self.assertEqual(7, len(hyps))
        data = [(0, 10.6), (1, 2), (1, -1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(7, len(hyps))
        data = [(0, 2), (1, 3), (1, 1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(8, len(hyps))
        data = [(0, -10.6), (1, 2), (1, 5.6)]
        hyps = list(rect_hypo(data))
        self.assertEqual(7, len(hyps))

    def test_rec_four_points(self):
        hyps = list(rect_hypo([(1., 1.), (2., 2.), (3., 0.), (4., 2.)]))
        self.assertEqual(14, len(hyps))
        data = [(1, 1), (2, 2), (1, 2), (2, 1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(10, len(hyps))
        data = [(1, 1), (1, 1), (1, 1), (1, 1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(2, len(hyps))
        data = [(0, 10.6), (1, 2), (1, -1), (0.5, 20)]
        hyps = list(rect_hypo(data))
        self.assertEqual(13, len(hyps))
        data = [(0, 10.6), (1, 2), (1, -1), (0.5, 10)]
        hyps = list(rect_hypo(data))
        self.assertEqual(11, len(hyps))
        data = [(0, 10.6), (1, 2), (1, -1), (0.5, 0)]
        hyps = list(rect_hypo(data))
        self.assertEqual(13, len(hyps))
        data = [(0, 10.6), (1, 2), (1, -1), (0.5, -2)]
        hyps = list(rect_hypo(data))
        self.assertEqual(12, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, 20)]
        hyps = list(rect_hypo(data))
        self.assertEqual(12, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, 21)]
        hyps = list(rect_hypo(data))
        self.assertEqual(14, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, 19)]
        hyps = list(rect_hypo(data))
        self.assertEqual(14, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, 9)]
        hyps = list(rect_hypo(data))
        self.assertEqual(14, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, 0)]
        hyps = list(rect_hypo(data))
        self.assertEqual(12, len(hyps))
        data = [(0, 10), (1, 20), (1, 0), (0.5, -5)]
        hyps = list(rect_hypo(data))
        self.assertEqual(14, len(hyps))
        data = [(0, -5), (1, 20), (1, 0), (0.5, 21)]
        hyps = list(rect_hypo(data))
        self.assertEqual(12, len(hyps))
        data = [(0, -5), (1, 20), (1, 0), (0.5, 20)]
        hyps = list(rect_hypo(data))
        self.assertEqual(11, len(hyps))
        data = [(0, -5), (1, 20), (1, 0), (0.5, 19)]
        hyps = list(rect_hypo(data))
        self.assertEqual(13, len(hyps))
        data = [(0, -5), (1, 20), (1, 0), (0.5, -1)]
        hyps = list(rect_hypo(data))
        self.assertEqual(11, len(hyps))
        data = [(0, -5), (1, 20), (1, 0), (0.5, 0)]
        hyps = list(rect_hypo(data))
        self.assertEqual(11, len(hyps))

    def test_rec_five_points(self):
        hyps = list(rect_hypo([(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]))
        self.assertEqual(19, len(hyps))

    def test_rec_six_points(self):
        data = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]
        hyps = list(rect_hypo(data))
        self.assertEqual(22, len(hyps))
        data = [(1, 2), (1, 3), (1, 4), (1, 5), (1, -6), (1, -2)]
        hyps = list(rect_hypo(data))
        self.assertEqual(22, len(hyps))

    def test_plane_single_point(self):
        data = [[(1, 1)], [(1, 0)], [(0, -5)], [(-1, 0)], [(-2.3, -5.34)]]
        for d in data:
            hyps = list(origin_plane_hypotheses(d))
            for pp in [[False], [True]]:
                self.assertTrue(assign_exists(d, hyps, pp))
            self.assertEqual(2, len(hyps))
        d = [(0, 0)]
        hyps = list(origin_plane_hypotheses(d))
        for pp in [[True]]:
            self.assertTrue(assign_exists(d, hyps, pp))
        self.assertEqual(1, len(hyps))

    def test_plane_two_points(self):
        data = [(-1, 0), (-1, 0)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, False],
                   [True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(2, len(hyps))
        print
        data = [(1, 1), (-2, -2)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, True],
                   [True, False],
                   [True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(3, len(hyps))
        print
        data = [(-1, 0), (-3, -2)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, True],
                   [True, False],
                   [False, False],
                   [True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(4, len(hyps))
        print
        data = [(-1, 0), (1, 0)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, True],
                   [True, False],
                   [True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(3, len(hyps))

    def test_plane_three_points(self):
        data = [(-1, 0), (-1, 0), (-1, 0)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, False, False],
                   [True, True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(2, len(hyps))
        print
        data = [(-1, 0), (-1, 0), (1, 0)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[False, False, True],
                   [True, True, False],
                   [True, True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(3, len(hyps))
        print
        data = [(1, 0), (-1, 0), (-4, -5)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[True, True, True],
                   [True, True, False],
                   [True, False, False],
                   [False, True, True],
                   [True, False, True],
                   [False, True, False]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(6, len(hyps))
        print
        data = [(1, 0), (1, 0), (-4, -5)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[True, True, True],
                   [True, True, False],
                   [False, False, True],
                   [False, False, False]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(4, len(hyps))
        print
        data = [(10, 1), (3, 3), (-1, -15)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[True, True, True],
                   [False, False, False],
                   [False, False, True],
                   [True, True, False],
                   [True, False, True],
                   [False, True, False]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(6, len(hyps))

    def test_plane_four_points(self):
        hyps = list(origin_plane_hypotheses(self._2d[4]))
        for pp in [[True, True, True, True],
                   [False, False, True, True],
                   [False, False, True, False],
                   [True, True, False, True],
                   [True, True, False, False],
                   [False, False, False, False]]:
            self.assertTrue(assign_exists(self._2d[4], hyps, pp))
        self.assertEqual(6, len(hyps))
        data = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        hyps = list(origin_plane_hypotheses(data))
        for pp in [[True, True, False, False],
                   [False, False, True, True],
                   [True, False, False, True],
                   [False, True, True, False],
                   [True, False, True, True],
                   [True, True, True, False],
                   [True, True, False, True],
                   [False, True, True, True]]:
            self.assertTrue(assign_exists(data, hyps, pp))
        self.assertEqual(8, len(hyps))

    def test_correlation(self):
        labels = [+1, +1, -1, +1]
        hyp = PlaneHypothesis(0, 1, 0)

        self.assertEqual(hyp.correlation(self._half_shatter, labels), -.5)

    def test_rad_estimate(self):
        self.assertAlmostEqual(1.0, rademacher_estimate(self._full_shatter,
                                                        self._hypotheses,
                                                        num_samples=1000,
                                                        random_seed=3),
                               places=1)

        self.assertAlmostEqual(0.0, rademacher_estimate([(0, 0)],
                                                        constant_hypotheses,
                                                        num_samples=1000,
                                                        random_seed=3),
                               places=1)

        self.assertAlmostEqual(0.625, rademacher_estimate(self._half_shatter,
                                                          self._hypotheses,
                                                          num_samples=1000,
                                                          random_seed=3),
                               places=1)

    def test_vc_one_point_pos(self):
        data_pos = [(1, False)]

        classifier_pos = train_sin_classifier(data_pos)

        for xx, yy in data_pos:
            self.assertEqual(True if yy == +1 else False,
                             classifier_pos.classify(xx))

    def test_vc_one_point_neg(self):
        data_neg = [(1, True)]

        classifier_neg = train_sin_classifier(data_neg)

        for xx, yy in data_neg:
            self.assertEqual(True if yy == +1 else False,
                             classifier_neg.classify(xx))

    def test_vc_three_points(self):
        data = [(1, False), (2, True), (3, False)]
        classifier = train_sin_classifier(data)

        for xx, yy in data:
            self.assertEqual(True if yy == +1 else False,
                             classifier.classify(xx))

    def test_vc_four_points(self):
        data = [(1, False), (2, True), (3, False), (5, False)]
        classifier = train_sin_classifier(data)

        for xx, yy in data:
            self.assertEqual(True if yy == +1 else False,
                             classifier.classify(xx))

    def test_plane_random(self):
        hyps = list(origin_plane_hypotheses(rand_data1))
        self.assertEqual(134, len(hyps))


def suite_rad():
    tests = ['test_correlation',
             'test_rad_estimate',
             'test_plane_single_point',
             'test_plane_two_points',
             'test_plane_three_points',
             'test_plane_four_points',
             'test_plane_random',
             'test_rec_single_point',
             'test_rec_two_points',
             'test_rec_three_points',
             'test_rec_four_points',
             'test_rec_five_points',
             'test_rec_six_points',
             'test_rec_random']
    return unittest.TestSuite(map(TestLearnability, tests))


if __name__ == '__main__':
    unittest.main(defaultTest='suite_rad')
