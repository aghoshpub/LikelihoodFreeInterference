#!/bin/bash   

#bash /sps/atlas/a/aghosh/batch/cpujobLaucher.sh /sps/atlas/a/aghosh/batch/runMadMinerDelphes.sh 0
for i in $(eval echo {1..$1})
do 
    #echo $i; 
    bash /sps/atlas/a/aghosh/batch/longcpujobLaucher.sh /sps/atlas/a/aghosh/batch/runMadMinerDelphes.sh $i
    sleep 0.5
done
