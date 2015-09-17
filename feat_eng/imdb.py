from csv import DictReader, DictWriter

__author__ = "Jianxiang Fan"
__email__ = "jianxiang.fan@colorado.edu"

import json
import re
import urllib2
import argparse
from bs4 import BeautifulSoup


def get_query_string(name):
    return '+'.join(re.findall('[A-Z][^A-Z]*', name))


def fetch_imdb_info(movie_name):
    resp = json.loads(urllib2.urlopen('http://www.omdbapi.com/?t=%s' % get_query_string(movie_name)).read())
    if resp['Response'] == 'True':
        return resp
    imdb_rearch = BeautifulSoup(
        urllib2.urlopen('http://www.imdb.com/find?q=%s' % movie_name).read())
    links = [td.a for td in imdb_rearch.findAll('td', {'class': 'result_text'})]
    for l in links:
        if re.sub('[^A-Za-z0-9]+', '', l.text.replace('&', 'and')).lower() == movie_name.lower():
            imdbid = l['href'].split('/')[2]
            resp = json.loads(urllib2.urlopen('http://www.omdbapi.com/?i=%s' % imdbid).read())
            if resp['Response'] == 'True':
                return resp
    print('%s not found' % movie_name)
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
