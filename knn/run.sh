#!/usr/bin/env bash

for ((i=3; i<=200; i=i+10))
do
    python knn.py --k=${i} --limit 1000 > 1.txt
done

for ((i=50; i<=2000; i=i+50))
do
    python knn.py --k=3 --limit ${i} > 2.txt
done
