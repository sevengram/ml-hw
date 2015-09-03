#!/usr/bin/env bash

for ((i=3; i<=200; i=i+10))
do
    echo k=${i} limit=1000
    python knn.py --k=${i} --limit 1000
done

for ((i=50; i<=2000; i=i+50))
do
    echo k=3 limit=${i}
    python knn.py --k=3 --limit ${i}
done
