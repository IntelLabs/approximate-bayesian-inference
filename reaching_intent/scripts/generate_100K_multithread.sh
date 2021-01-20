#!/usr/bin/env bash
python3.7 main_datageneration.py datasets/test0.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test1.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test2.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test3.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test4.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test5.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test6.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test7.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test8.dat 10000 1000 &
python3.7 main_datageneration.py datasets/test9.dat 10000 1000 &

wait

cat datasets/test0.dat >> $1
cat datasets/test1.dat >> $1
cat datasets/test2.dat >> $1
cat datasets/test3.dat >> $1
cat datasets/test4.dat >> $1
cat datasets/test5.dat >> $1
cat datasets/test6.dat >> $1
cat datasets/test7.dat >> $1
cat datasets/test8.dat >> $1
cat datasets/test9.dat >> $1

rm datasets/test*.dat

