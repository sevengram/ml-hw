#!/usr/bin/env python2

import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    f = open("./report/report.txt")
    group = [[], []]
    i = 0
    while True:
        line = f.readline()
        if not line:
            break
        if not line.strip(' \r\n\t'):
            continue
        if line[0] == 'k':
            k, limit = tuple([int(s.split('=')[1]) for s in line.split()])
        elif line[0] == 'A':
            accuracy = float(line.split()[1])
            group[i].append((k, limit, accuracy))
        elif line[0] == '=':
            i = 1
    f.close()

    d1 = group[0]
    xdata = [t[0] for t in d1]
    ydata = [t[2] for t in d1]
    pyplot.plot(xdata, ydata, 'b*')
    pyplot.plot(xdata, ydata, 'r')
    pyplot.xlabel('K')
    pyplot.ylabel('Accuracy')
    pyplot.ylim(0, 1)
    pyplot.xticks(range(0, group[0][-1][0] + 20, 20))
    pyplot.yticks(numpy.arange(0.1, 1.1, 0.1))
    pyplot.title('Fig 1. K - Accuracy (Data limit: 2000)')
    pyplot.show()

    d1 = group[1]
    xdata = [t[1] for t in d1]
    ydata = [t[2] for t in d1]
    pyplot.plot(xdata, ydata, 'b*')
    pyplot.plot(xdata, ydata, 'r')
    pyplot.xlabel('Size of Training Set')
    pyplot.ylabel('Accuracy')
    pyplot.ylim(0, 1)
    pyplot.xticks(range(0, group[1][-1][1] + 250, 250))
    pyplot.yticks(numpy.arange(0.1, 1.1, 0.1))
    pyplot.title('Fig 2. Size of Training Set - Accuracy (K=3)')
    pyplot.show()
