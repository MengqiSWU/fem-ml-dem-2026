#!/bin/bash
# python 
alias python="/usr/bin/python3"

# yade & escript -> tongming
export PYTHONPATH=$PYTHONPATH:/home/tongming/fem-ml-dem:/home/tongming/fem-ml-dem/FEMxML:/home/tongming/fem-ml-dem/FEMxDEM:/home/tongming/fem-ml-dem/sciptes4figures:/home/tongming/fem-ml-dem/FEMxEPxML:/home/tongming/escript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tongming/escript/lib:/home/tongming/yade/install/lib/x86_64-linux-gnu/yade-2022-08-01.git-f890b06

echo ==== START ====

#python3 footing_mengqi.py -numg_net 254 -num_mesh 303
#python3 footing_mengqi.py -numg_net 480 -num_mesh 303
#python3 footing_mengqi.py -numg_net 3114 -num_mesh 303

#python3 footing_mengqi.py -numg_net 254 -num_mesh 552
#python3 footing_mengqi.py -numg_net 480 -num_mesh 552
#python3 footing_mengqi.py -numg_net 3114 -num_mesh 552

#python3 footing_mengqi.py -numg_net 254 -num_mesh 1206
#python3 footing_mengqi.py -numg_net 480 -num_mesh 1206
#python3 footing_mengqi.py -numg_net 3114 -num_mesh 1206

# footing ratio sampling
#python3 footing_mengqi.py -numg_net 3114 -num_mesh 1206 -ratio 0.1 -integration_order 1
#python3 footing_mengqi.py -numg_net 3114 -num_mesh 1206 -ratio 1.0 -integration_order 1

# active learning
python3 footing_tongming.py -numg_net 3114 -num_mesh 1206 -ratio 1.0 -integration_order 1 -active_flag 1 -active_iter 4

echo ====  FINISH  ===
