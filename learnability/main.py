#!/usr/bin/env python2

from rademacher import OriginPlaneHypothesis, coin_tosses

if __name__ == '__main__':
    hypo = OriginPlaneHypothesis(1, -1)
    print(hypo.classify((1, 1)))
    print(coin_tosses(1, 3))
