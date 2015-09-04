#!/usr/bin/env bash

for ((i=3; i<=200; i=i+5))
do
    echo k=${i} limit=2000
    python knn.py --k=${i} --limit 2000
done

for ((i=50; i<=4000; i=i+100))
do
    echo k=3 limit=${i}
    python knn.py --k=3 --limit ${i}
done
