#!/bin/bash
# python 
alias python="/usr/bin/python3"

# yade & escript -> tongming
export PYTHONPATH=$PYTHONPATH:/home/tongming/fem-ml-dem:/home/tongming/fem-ml-dem/FEMxML:/home/tongming/fem-ml-dem/FEMxDEM:/home/tongming/fem-ml-dem/sciptes4figures:/home/tongming/fem-ml-dem/FEMxEPxML:/home/tongming/escript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tongming/escript/lib:/home/tongming/yade/install/lib/x86_64-linux-gnu/yade-2022-08-01.git-f890b06

echo ==== START ====

for (( i = 6; i < 7; i++ )); do
cd FEMxML
python3 torch_main_biax_1e5.py -n $i
cd ..
python3 explicit_biaxial.py -n $i
python3 explicit_retaining_wall.py -n $i
python3 explicit_strip_footing.py -n $i

done

#python3 explicit_biaxial.py -mode mixedclear
#python3 explicit_biaxial.py -mode mldem

echo ====  FINISH  ===
