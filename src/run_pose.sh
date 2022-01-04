#!/bin/bash

for i in {1..51}
do
   echo $i
   k=$(($i+1))
   echo $k
   ./HelloWorld poses/pose_${i}.xyz 350 0.000001 0.2 poses/pose_${k}.xyz 
done