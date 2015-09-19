__author__ = "Jianxiang Fan"
__email__ = "jianxiang.fan@colorado.edu"

import time
import argparse
import itertools
from csv import DictReader, DictWriter

import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, VectorizerMixin
from sklearn.linear_model import SGDClassifier

kPAGE_FIELD = 'page'
kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

first_person_words = ['i', 'we', 'you', 'us', 'my', 'mine', 'your', 'yours']
third_person_words = ['he', 'she', 'they', 'their', 'his', 'him', 'her', 'theirs', 'hers']


class SentenceMixin(VectorizerMixin):
    def build_analyzer(self):
        """
        Return a callable that handles preprocessing and tokenization
        """
        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()
        stemmer = LancasterStemmer()

        filter_meta = lambda doc: ' '.join([w for w in doc.split() if not w.startswith('~')])
        parse_words = lambda doc: tokenize(preprocess(filter_meta(self.decode(doc))))
        stem_words = lambda doc: [stemmer.stem(t) for t in parse_words(doc)]
        meta_func = lambda prefix: lambda doc: (t for t in self.decode(doc).split() if t.startswith(prefix))

        feat_func_map = {
            'word': lambda doc: self._word_ngrams(parse_words(doc), self.get_stop_words()),
            'stem': lambda doc: self._word_ngrams(stem_words(doc), self.get_stop_words()),
            '1st': lambda doc: ('~T:1st' for i in parse_words(doc) if i in first_person_words),
            '3rd': lambda doc: ('~T:3rd' for i in parse_words(doc) if i in third_person_words),
            'tag': lambda doc: self._word_ngrams([t[1] for t in nltk.pos_tag(parse_words(doc))]),
            'length': lambda doc: ['~L:%d' % (len(parse_words(doc)) / 5)],
            'genre': meta_func('~G'),
            'rating': meta_func('~Ra'),
            'votes': meta_func('~V'),
            'lang': meta_func('~La'),
            'country': meta_func('~Co'),
            'year': meta_func('~Y'),
            'runtime': meta_func('~Rt'),
            'type': meta_func('~T')
        }
        func_list = [feat_func_map.get(flag.strip()) for flag in self.analyzer.split(':')] \
            if type(self.analyzer) is str else None
        if not func_list:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)
        else:
            return lambda doc: itertools.chain.from_iterable(f(doc) for f in func_list if callable(f))


class SentenceVectorizer(CountVectorizer, SentenceMixin):
    pass


class TfidfSentenceVectorizer(TfidfVectorizer, SentenceMixin):
    pass


def create_documents(examples, metainfo=None):
    for e in examples:
        doc = e[kTEXT_FIELD]
        page = e[kPAGE_FIELD]
        if metainfo and page in metainfo:
            for v in metainfo[page]['genre']:
                if v.upper() != 'N/A':
                    doc += " ~G:" + v
            for v in metainfo[page]['lang']:
                if v.upper() != 'N/A':
                    doc += " ~La:" + v
            for v in metainfo[page]['country']:
                if v.upper() != 'N/A':
                    doc += " ~Co:" + v
            if metainfo[page]['type'].upper() != 'N/A':
                doc += " ~T:%s" % metainfo[page]['type']
            if metainfo[page]['rating']:
                doc += " ~Ra:%s" % metainfo[page]['rating']
            if metainfo[page]['votes']:
                doc += " ~V:%s" % metainfo[page]['votes']
            if metainfo[page]['year']:
                doc += " ~Y:%s" % metainfo[page]['year']
            if metainfo[page]['runtime']:
                doc += " ~Rt:%s" % metainfo[page]['runtime']
        yield doc


class Featurizer:
    def __init__(self, min_n, max_n, analyzer, use_tfidf=False, stop_words=False):
        self.vectorizer = SentenceVectorizer(analyzer=analyzer, ngram_range=(min_n, max_n),
                                             stop_words='english' if stop_words else None) if not use_tfidf else \
            TfidfSentenceVectorizer(analyzer=analyzer, ngram_range=(min_n, max_n),
                                    stop_words='english' if stop_words else None)

    def train_feature(self, examples, metainfo=None):
        return self.vectorizer.fit_transform(create_documents(examples, metainfo))

    def test_feature(self, examples, metainfo=None):
        return self.vectorizer.transform(create_documents(examples, metainfo))

    def show_top(self, classifier, categories, n):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top = np.argsort(classifier.coef_[0])[-n:][::-1]
            bottom = np.argsort(classifier.coef_[0])[:n]
            print('Pos: %s' % ','.join(feature_names[top]))
            print('Neg: %s' % ','.join(feature_names[bottom]))
        else:
            for i, category in enumerate(categories):
                top = np.argsort(classifier.coef_[i])[-n:]
                print('%s: %s' % (category, ' '.join(feature_names[top])))


def load_meta_info(meta_file):
    meta = {}
    all_votes = []
    split_group = lambda di, key: di.get(key, '').replace(' ', '').split(',')
    meta_raw = list(DictReader(open(meta_file, 'r')))
    for m in meta_raw:
        meta[m['page']] = {}
        rating = m.get('imdbRating', 'N/A')
        release = m.get('Released', 'N/A')
        runtime = m.get('Runtime', 'N/A')
        votes = str(m.get('imdbVotes', 'N/A')).replace(',', '')
        if votes != 'N/A':
            all_votes.append(int(votes))
        meta[m['page']]['genre'] = split_group(m, 'Genre')
        meta[m['page']]['lang'] = split_group(m, 'Language')
        meta[m['page']]['country'] = split_group(m, 'Country')
        meta[m['page']]['type'] = m.get('Type', 'N/A')
        meta[m['page']]['year'] = int(release.split()[-1]) / 20 * 20 if release != 'N/A' else None
        meta[m['page']]['runtime'] = int(runtime.split()[0]) / 30 + 1 if runtime != 'N/A' else None
        meta[m['page']]['rating'] = int(float(rating)) + 1 if rating != 'N/A' else None
        meta[m['page']]['votes'] = int(votes) if votes != 'N/A' else None
    all_votes.sort()
    score_votes = lambda s: all_votes.index(s) / (len(all_votes) / 10)
    for p, m in meta.iteritems():
        real_votes = m['votes']
        m['votes'] = score_votes(real_votes) + 1 if real_votes else None
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Output file',
                        type=str, default='predictions.csv', required=False)
    parser.add_argument('--train', help='Training set file',
                        type=str, default='../data/spoilers/train.csv', required=False)
    parser.add_argument('--test', help='Test set file',
                        type=str, default='../data/spoilers/test.csv', required=False)
    parser.add_argument('--meta', help='Meta info file',
                        type=str, default='./meta_info.csv', required=False)
    parser.add_argument('--cvt', help='Number of times do random cross validation, default 0',
                        type=int, default=0, required=False)
    parser.add_argument('--cvp', help='Proportion of dataset to use as test in cross validation',
                        type=float, default=0.25, required=False)
    parser.add_argument('--top', help='Number of top features to show',
                        type=int, default=50, required=False)
    parser.add_argument('--feat', help='Features concat by \':\', default is \'word\', support feats: word:stem:1st:'
                                       '3rd:length:tag:rating:votes:year:runtime:genre:country:lang:type',
                        type=str, default='word', required=False)
    parser.add_argument('--ngmin', help='The lower boundary of the range of n-values for n-grams',
                        type=int, default=1, required=False)
    parser.add_argument('--ngmax', help='The upper boundary of the range of n-values for n-grams',
                        type=int, default=1, required=False)
    parser.add_argument("--tfidf", help="Use tf-idf", action='store_true')
    parser.add_argument("--stop", help="Use stop words", action='store_true')
    args = parser.parse_args()

    # Load meta data
    meta_info = load_meta_info(args.meta) if args.meta else None
    print('Meta info loaded!')

    # Cast to list to keep it all in memory
    train = list(DictReader(open(args.train, 'r')))
    test = list(DictReader(open(args.test, 'r')))
    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])
    print('Text data loaded!')

    feat = Featurizer(args.ngmin, args.ngmax, args.feat, args.tfidf, args.stop)
    x_train = feat.train_feature(train, meta_info)
    y_train = np.array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    # Train classifier and do cross validation
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    if args.cvt:
        print("Start cross validation...")
        cv = ShuffleSplit(len(y_train), args.cvt, test_size=args.cvp, random_state=int(time.time()))
        scores = cross_val_score(lr, x_train, y_train, cv=cv, scoring='accuracy', verbose=1)
        print(scores)
        print(scores.mean(), scores.std())

    # Run test data
    print("Run test set...")
    x_test = feat.test_feature(test, meta_info)
    lr.fit(x_train, y_train)
    feat.show_top(lr, labels, args.top)
    predictions = lr.predict(x_test)
    o = DictWriter(open(args.output, 'w'), ['id', 'spoiler'])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
    print 'Finish!'
