#!/usr/bin/env python2

from rademacher import OriginPlaneHypothesis

if __name__ == '__main__':
    hypo = OriginPlaneHypothesis(1, -1)
    print(hypo.classify((1, 1)))
