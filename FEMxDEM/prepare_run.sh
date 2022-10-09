#!/bin/bash
# python 
alias python="/usr/bin/python3"

# yade
#export PATH=$PATH:/home/shguan/yade/install/bin
#export PYTHONPATH=$PYTHONPATH:/home/shguan/fem-ml-dem-remote/FEMxDEM
#export PYTHONPATH=$PYTHONPATH:/home/shguan/fem-ml-dem-remote/FEMxML
#export PYTHONPATH=$PYTHONPATH:/home/shguan/fem-ml-dem-remote
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shguan/yade/install/lib/x86_64-linux-gnu/yade-2021-04-07.git-fed3d41

# escript
#export PYTHONPATH=$PYTHONPATH:/home/shguan/escript/src
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shguan/escript/src/lib

# yade & escript -> tongming
export PYTHONPATH=$PYTHONPATH:/home/mengqi/fem-ml-dem:/home/mengqi/fem-ml-dem/FEMxDEM:/home/mengqi/fem-ml-dem/FEMxML:/home/mengqi/fem-ml-dem/FEMxEPxML:/home/mengqi/escript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengqi/escript/lib:/home/mengqi/yade/install/lib/x86_64-linux-gnu/yade

echo ==== START ====

python3 prepareRVE.py
#python3 explicit_biaxial.py -n 5
#python3 explicit_retaining_wall.py -n 5
#python3 explicit_strip_footing.py -n 5

#python3 explicit_biaxial.py -mode mixed
#python3 explicit_biaxial.py -mode mldem

echo ====  FINISH  ===
