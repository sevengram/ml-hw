__author__ = "Jianxiang Fan"
__email__ = "jianxiang.fan@colorado.edu"

import sys
import re
import json
import urllib2
import argparse
from csv import DictReader, DictWriter

from bs4 import BeautifulSoup


def get_query_string(name):
    name = name.replace('Remake', '').replace('Reimagined', '')
    name = re.sub('[0-9]+$', '', name)
    return '+'.join(re.findall('[A-Z][^A-Z]*', name))


def almost_same(n1, n2):
    if not n1 or not n2:
        return False
    n1 = n1.lower()
    n2 = n2.lower()
    n1_first_half = n1[:(len(n1) + 1) / 2]
    n2_first_half = n2[:(len(n2) + 1) / 2]
    n1_second_half = n1[len(n1) / 2:]
    n2_second_half = n2[len(n2) / 2:]
    return n1 == n2 or n1.startswith(n2_first_half) or n2.startswith(n1_first_half) or n1.endswith(
        n2_second_half) or n2.endswith(n1_second_half)


def fetch_imdb_info(movie_name):
    resp = json.loads(urllib2.urlopen('http://www.omdbapi.com/?t=%s' % get_query_string(movie_name)).read())
    if resp['Response'] == 'True':
        return resp
    imdb_rearch = BeautifulSoup(
        urllib2.urlopen('http://www.imdb.com/find?q=%s' % get_query_string(movie_name)).read(), "html5lib")
    links = [(td.a, td.i) for td in imdb_rearch.findAll('td', {'class': 'result_text'})]
    for la, li in links:
        imdb_name = re.sub('[^A-Za-z0-9]+', '', la.text.replace('&', 'and'))
        aka_name = re.sub('[^A-Za-z0-9]+', '', li.text) if li else ''
        if almost_same(imdb_name, movie_name) or almost_same(aka_name, movie_name):
            imdbid = la['href'].split('/')[2]
            resp = json.loads(urllib2.urlopen('http://www.omdbapi.com/?i=%s' % imdbid).read())
            if resp['Response'] == 'True':
                return resp
    print('%s not found' % movie_name)
    sys.stdout.flush()
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Output file',
                        type=str, default='info.csv', required=False)
    parser.add_argument('--train', help='Training set file',
                        type=str, default='../data/spoilers/train.csv', required=False)
    parser.add_argument('--test', help='Test set file',
                        type=str, default='../data/spoilers/test.csv', required=False)
    args = parser.parse_args()

    fields = ['page', 'Country', 'Genre', 'Released', 'imdbRating', 'Runtime', 'Language', 'Type', 'imdbVotes']

    filter_dict = lambda d, r: {ki: vi for (ki, vi) in d.iteritems() if ki in r}

    full_set = list(DictReader(open(args.train, 'r'))) + list(DictReader(open(args.test, 'r')))
    dict_result = {}
    for x in full_set:
        movie = x['page']
        if movie not in dict_result:
            imdb_info = fetch_imdb_info(movie)
            dict_result[movie] = filter_dict(imdb_info, fields) if imdb_info else None
    with open(args.output, 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for k, v in dict_result.iteritems():
            if v:
                writer.writerow(dict(v, page=k))
