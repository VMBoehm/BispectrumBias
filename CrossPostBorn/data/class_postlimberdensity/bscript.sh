#!/bin/bash


for ((i=100;i<500;i+=50))
do
    j=$((i+20))    
    echo $i $j

    k=$(printf "%03d" $i)
    l=$(printf "%03d" $j)
    #echo $k $l
    echo $k $l >> logbscriptv2.txt

    str=./pborncross/inifiles2/class_z${k}z${l}v2_inparameters.ini
    #echo $str 
    echo $str >> logbscriptv2.txt

    time  ./class $str >> logbscriptv2.txt
done
