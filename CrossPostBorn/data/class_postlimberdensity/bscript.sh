#!/bin/bash

#for i in {0..10..1}
#do
#    echo $i
#done
#
#

##num=$(awk 'BEGIN{for(i=0;i<=1;i+=0.1)print i}')
#num=$(python -c "print(i for i in range(10))")
#echo $num
#
#n=1
#for ((k=400;k<420;k+=2))
#do
#    i=$(bc <<<"scale=2; $k / 100" )    # when k=402 you get i=4.02, etc
#    echo $i
#    j=$((k+2)) 
#    l=$(bc <<<"scale=2; $j / 100" )    # when k=402 you get i=4.02, etc
#    echo $l
#    vals=($(seq -s, $k 2 $j))
#    echo $vals
#    vals=($(seq -s, 0.1 0.2 1))
#    echo $vals
#    vals=($(seq -s, $i 0.2 $l))
#    echo $vals
#done
#
for ((i=000;i<100;i+=10))
do
    j=$((i+20))    
    echo $i $j

    k=$(printf "%03d" $i)
    l=$(printf "%03d" $j)
    echo $k $l

    str=inifiles/class_z${k}z${l}_inparameters.ini
    echo $str
    #srun ./class $str pk_ref'
done





