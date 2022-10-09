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
export PYTHONPATH=$PYTHONPATH:/home/tongming/fem-ml-dem:/home/tongming/fem-ml-dem/FEMxML:/home/tongming/fem-ml-dem/FEMxDEM:/home/tongming/fem-ml-dem/sciptes4figures:/home/tongming/fem-ml-dem/FEMxEPxML:/home/tongming/escript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tongming/escript/lib:/home/tongming/yade/install/lib/x86_64-linux-gnu/yade-2022-08-01.git-f890b06

echo ==== START ====

python3 ./FEMxML/torch_main_biax_1e5.py

echo ====  FINISH  ===
