#!/bin/bash
#SBATCH -J tri_mp
#SBATCH -n 8

/usr/bin/mpirun -n 8 python3 triaxialRough.py
