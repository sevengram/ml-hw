#!/usr/bin/env python2
from random import randint

from rademacher import origin_plane_hypotheses

if __name__ == '__main__':
    # t1 = time.time()
    # count = 0
    # for i in axis_aligned_hypotheses([(randint(0, 1000), randint(0, 1000)) for i in range(100)]):
    #     count += 1
    data = [(randint(-50,50),randint(-50,50)) for i in range(20)]
    print data
    hyps = list(origin_plane_hypotheses(data))
    print(len(hyps))
    # print(time.time() - t1)
    # print(count)
