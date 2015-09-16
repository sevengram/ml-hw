__author__ = "Jianxiang Fan"
__email__ = "jianxiang.fan@colorado.edu"

import argparse
from csv import DictReader, DictWriter
import itertools

import nltk
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

first_person_words = ['i', 'we', 'you', 'us', 'my', 'mine', 'your', 'yours']

third_person_words = ['he', 'she', 'they', 'their', 'his', 'him', 'her', 'theirs', 'hers']


class DiscussVectorizer(CountVectorizer):
    def build_analyzer(self):
        """
        Return a callable that handles preprocessing and tokenization
        """
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()
        pre_token = lambda doc: tokenize(preprocess(self.decode(doc)))

        feature_map = {
            'length': lambda doc: ['L:%d' % (len(pre_token(doc)) / 5)],
            '1st': lambda doc: ('T:1st' for i in pre_token(doc) if i in first_person_words),
            '3rd': lambda doc: ('T:3rd' for i in pre_token(doc) if i in third_person_words),
            'word': lambda doc: self._word_ngrams(pre_token(doc), self.get_stop_words()),
            'tag': lambda doc: self._word_ngrams([t[1] for t in nltk.pos_tag(pre_token(doc))])
        }

        func_list = None
        if type(self.analyzer) is str:
            func_list = [feature_map[flag.strip()] for flag in self.analyzer.split('|') if flag.strip() in feature_map]
        if not func_list:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)
        else:
            return lambda doc: itertools.chain.from_iterable(f(doc) for f in func_list)


class Featurizer:
    def __init__(self, min_n, max_n):
        self.vectorizer = DiscussVectorizer(analyzer="word|tag", ngram_range=(min_n, max_n))
        # self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(min_n, max_n))

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top(self, classifier, categories, n):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top = np.argsort(classifier.coef_[0])[-n:]
            bottom = np.argsort(classifier.coef_[0])[:n]
            print('Pos: %s' % ','.join(feature_names[top]))
            print('Neg: %s' % ','.join(feature_names[bottom]))
        else:
            for i, category in enumerate(categories):
                top = np.argsort(classifier.coef_[i])[-n:]
                print('%s: %s' % (category, ' '.join(feature_names[top])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', help='Run cross validation after training', action='store_true')
    parser.add_argument('--cv', help='Number of folds in cross validation',
                        type=int, default=10, required=False)
    parser.add_argument('--top', help='Number of top features to show',
                        type=int, default=20, required=False)
    parser.add_argument('--output', help='Output file',
                        type=str, default='predictions.csv', required=False)
    parser.add_argument('--train', help='Training set file',
                        type=str, default='../data/spoilers/train.csv', required=False)
    parser.add_argument('--test', help='Test set file',
                        type=str, default='../data/spoilers/test.csv', required=False)
    parser.add_argument('--ngmin', help='The lower boundary of the range of n-values for n-grams',
                        type=int, default=1, required=False)
    parser.add_argument('--ngmax', help='The upper boundary of the range of n-values for n-grams',
                        type=int, default=1, required=False)
    args = parser.parse_args()

    # Cast to list to keep it all in memory
    train = list(DictReader(open(args.train, 'r')))
    test = list(DictReader(open(args.test, 'r')))
    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    feat = Featurizer(args.ngmin, args.ngmax)
    print('Label set: %s' % str(labels))
    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    y_train = np.array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    # Train classifier & make cross validation
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    if args.val:
        scores = cross_val_score(lr, x_train, y_train, cv=args.cv, scoring='accuracy')
        print(scores)
        print(scores.mean())
        print(scores.std())
    else:
        x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
        lr.fit(x_train, y_train)
        feat.show_top(lr, labels, args.top)
        predictions = lr.predict(x_test)
        o = DictWriter(open(args.output, 'w'), ['id', 'spoiler'])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in test], predictions):
            d = {'id': ii, 'spoiler': labels[pp]}
            o.writerow(d)
