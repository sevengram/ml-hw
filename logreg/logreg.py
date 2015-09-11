__author__ = "Jianxiang Fan"
__email__ = "jianxiang.fan@colorado.edu"

import argparse
import random
import time
from math import exp, log
from collections import defaultdict

import numpy

kSEED = time.time()
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * numpy.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """

    def __init__(self, label, words, vocab, idf):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        :param idf: Inverse document frequency of the words
        """
        self.nonzero = {}
        self.y = label
        self.x = numpy.zeros(len(vocab))
        self.tfidf = numpy.zeros(len(vocab))
        total_wc = 0
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
                total_wc += float(count)
        self.x[0] = 1
        if idf:
            for k, v in self.nonzero.iteritems():
                tf = self.x[k] / total_wc
                self.tfidf[k] = tf * idf[k]
            self.tfidf[0] = 1


class LogReg:
    def __init__(self, num_features, mu, step=lambda x: 0.05):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        self.beta = numpy.zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = defaultdict(int)
        self.marker = [0] * num_features
        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(numpy.dot(self.beta, ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        x = train_example.x if not use_tfidf else train_example.tfidf
        h = lambda theta, v: sigmoid(numpy.dot(theta, v))
        delta = train_example.y - h(self.beta, x)
        if self.mu == 0.0:
            self.beta += self.step(iteration) * delta * x
        else:
            for ii in xrange(len(self.beta)):
                if x[ii] != 0:
                    self.beta[ii] = (self.beta[ii] + self.step(iteration) * delta * x[ii]) * \
                                    (1 - 2 * self.step(iteration) * self.mu) ** (self.marker[ii] + 1)
                    self.marker[ii] = 0
                else:
                    self.marker[ii] += 1
        return self.beta

    def sg_update2(self, train_example, iteration, use_tfidf=False):
        x = train_example.x if not use_tfidf else train_example.tfidf
        h = lambda theta, v: sigmoid(numpy.dot(theta, v))
        delta = train_example.y - h(self.beta, x)
        self.beta[0] += self.step(iteration) * delta * x[0]
        self.beta[1:] = \
            (1 - self.step(iteration) * self.mu) * self.beta[1:] + self.step(iteration) * delta * x[1:]
        return self.beta

    def significant_features(self, count=20, postive=True):
        return numpy.argsort(self.beta)[-count:] if postive else numpy.argsort(self.beta)[:count]

    def nonsignificant_features(self, count=20):
        return numpy.argsort(numpy.abs(self.beta))[:count]


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proportion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vl = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vl[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vl[0]

    doc_count = 0
    for filepath in [positive, negative]:
        doc_count += sum(1 for line in open(filepath))
    idf = [log(float(doc_count) / (f + 1.0)) for f in df]
    train_set = []
    test_set = []
    for label, filepath in [(1, positive), (0, negative)]:
        for line in open(filepath):
            ex = Example(label, line.split(), vl, idf)
            if random.random() <= test_proportion:
                test_set.append(ex)
            else:
                train_set.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set, vl


def step_update(iteration):
    """
    Provide an effective iteration dependent step size

    :param iteration: The current iteration (an integer)
    :return: Return the step size
    """
    return 0.1 / (1 + 0.1 * 0.05 * iteration)


def step_update_build(initial_step, alpha):
    """
    Create a step update function based on parameters

    :param initial_step: Initial SG step size
    :param alpha: Parameter for step update
    :return: Return the step update function
    """
    return lambda itr: initial_step / (1 + initial_step * alpha * itr)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    # Extra arguments
    argparser.add_argument("--alpha", help="Parameter for step update",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--tfidf", help="Use tf-idf", action='store_true')

    args = argparser.parse_args()

    train, test, vocab_list = read_dataset(args.positive, args.negative, args.vocab)
    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab_list), args.mu, step_update_build(args.step, args.alpha))

    # Iterations
    update_number = 0
    for pp in xrange(args.passes):
        random.shuffle(train)
        for i in train:
            update_number += 1
            lr.sg_update(i, update_number, use_tfidf=args.tfidf)
            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))

    postive_words = lr.significant_features(postive=True)
    print([vocab_list[i] for i in postive_words])
    negative_words = lr.significant_features(postive=False)
    print([vocab_list[i] for i in negative_words])
    nonsignificant_words = lr.nonsignificant_features()
    print([vocab_list[i] for i in nonsignificant_words])
