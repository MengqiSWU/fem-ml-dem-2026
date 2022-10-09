""" Author: Ning Guo <ceguo@connect.ust.hk>
    run `mv biaxialSmooth.yade.gz 0.yade.gz`
    to generate initial RVE packing

How to run this script:
    sudo apt install python3-escript
    cd examples/FEMxDEM
    export PYTHONPATH="/usr/lib/python3-escript:../../py/FEMxDEM"
    export LD_LIBRARY_PATH=/usr/lib/python3-escript/lib
    ln -s /path/to/yade ../../py/FEMxDEM/yadeimport.py
    /path/to/yade ./biaxialSmooth.py
Please amend these instructions if you find that they do not work.
"""
# from PyQt5 import QtWebKit, QtWebKitWidgets  #  uninstall the PySide2 OR THE MODULU CAN NOT BE IMPORT DURING DEBUG

import sys
# from network import RestoreNet
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from esys.escript import util
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner
from esys.finley import Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
from utilSelf.saveGauss import saveGauss2D
from FEMxDEM.msFEM2DLast import MultiScale
from utilSelf.girdPlot import gridPlot
import time
import os
import errno
from matplotlib import pyplot as plt
from utilSelf.general import mkdirsSelf

# from FEMxML.netTorch import Net
from FEMxML.netTorchLastDouble import Net
# from FEMxML.netTorchRotate import Net
# from FEMxML.netTorchStress import NetStress
# from FEMxML.netTorchTangent import NetTangent

# read the input values
# argList = sys.argv
# argc = len(argList)
# n = 0
# useml = False
# while n < argc:
#     if argList[n][:2] == "-m": # calculation mode, 0 for multiscale, 1 for machine learning
#         n += 1
#         useml = (1 == int(argList[n]))
#     n += 1


def get_pool(mpi=False, threads=1):
    """ function to return pool for parallelization
        supporting both MPI (experimental) on distributed
        memory and multiprocessing on shared memory.
    """
    if mpi:  # using MPI
        from FEMxDEM.mpipool import MPIPool
        pool = MPIPool()
        pool.start()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads)
    else:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool

useMPI = False
useml = True  # True or False
nump = 6  # number of processes for multiprocessing
mpi_pool = get_pool(mpi=useMPI, threads=8) if useMPI else None
loadingPath = 'biaxial'  # 'biaxial' 'confinedCompression' 'gaussianConfinedPressure'
mode = 'net'  # 'net' or 'find'
numRandom = 4
confining = -1.e5  # confining pressure
lx = 0.05
ly = 0.1  # sample size, 50mm by 100mm
axialStrain = -0.1
# NOTE: if the velocity too bigger than the training samples, then the computation will not converge
# vel = ly*axialStrain/ly  # loading velocity
vel = -0.0001  # -0.0001 default
loadSubStep = int(abs(ly * axialStrain / vel)) + 1
# --------------------------Mesh size----------------------------
# nList = [2, 4]
# nList = [4, 8]
# nList = [6, 12]
nList = [8, 16]
# nList = [12, 24]
# nList = [10, 20]
nx = nList[0]
ny = nList[1]  # sample discretization, 8 by 16 quadrilateral elements

rtol = 1e-2 if useml else 1e-2

# create the simulation directory
mkdirsSelf('../simu')

# load informationsque
loadInfor = ('../simu/ML_%s' % mode if useml else '../simu/Right_DEM') + '_' + str(nx) + '_' + str(
    ny) + '_' + loadingPath + (
                str(numRandom) if 'gaussianConfinedPressure' in loadingPath else '')

print('-' * 80)
print('\tML mode: %s' % mode if useml else '\tDEM-FEM')
print('lx:\t%.5f' % lx + '\t nx:\t%d' % nx)
print('ly:\t%.5f' % ly + '\t ny:\t%d' % ny)
print('confing:\t%e' % confining)
print('loading velocity:\t%e' % vel)
print('err tolerance:\t%e' % rtol)
print('Axial strain:\t%f' % axialStrain + '\t NumSubstep:\t%d' % loadSubStep)
print(loadInfor)
print('-' * 80)

# mkdirs
mkdirsSelf(
    loadInfor,
    os.path.join(loadInfor, 'gauss'),
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'packing'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_vtk'),
    os.path.join(loadInfor, 'iteration_packing'))

# mydomain 相当于一个有限元模型或者拉格朗日插值空间
mydomain = Rectangle(l0=lx, l1=ly, n0=nx, n1=ny,
                     order=2, integrationOrder=2)
dim = mydomain.getDim()
k = kronecker(mydomain)
numg = 4 * nx * ny  # number of Gauss points, 4 GP each element (reduced integration)

x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()
# plot the node & boundary-node
if ny * ny <= 16:
    grid = gridPlot(nx, ny, x, bx, loadInfor)
    grid.plot()

# prob 为DEM计算模型
prob = MultiScale(domain=mydomain, ng=numg, np=nump, random=False, useMPI=useMPI, mpi_pool=mpi_pool,
                  rtol=rtol, usePert=False, pert=-2.e-5, verbose=True,
                  useml=useml, loadInfor=loadInfor, mode=mode)

disp = Vector(0., Solution(mydomain))

# disp_check = disp.toListOfTuples()
t = 0

stress = prob.getCurrentStress()  # initial stress
proj = Projector(mydomain)
sig = proj(stress)  # project Gauss point value to nodal value
sig_bounda = interpolate(sig, FunctionOnBoundary(mydomain))  # interpolate
traction = matrix_mult(sig_bounda, mydomain.getNormal())  # boundary traction

# temp1_check = sup(bx[1]).toListOfTuples()
topSurf = whereZero(bx[1] - sup(bx[1]))  # equals 1 if the node is on the top
tractTop = traction * topSurf  # traction at top surface
forceTop = integrate(tractTop, where=FunctionOnBoundary(mydomain))  # resultant force at top
lengthTop = integrate(topSurf, where=FunctionOnBoundary(mydomain))  # length of top surface
# forceTop_check = forceTop.toListOfTuples()
# lengthTop_check = lengthTop.toListOfTuples()
fout = open(os.path.join(loadInfor, 'biaxial_surf.dat'), 'w')
fout.write('0 ' + str(forceTop[1]) + ' ' + str(lengthTop) + ' ' + str(1.) + '\n')
fout.close()

# get initial boundary
leftBoundary = whereZero(bx[0])
rightBoundary = whereZero(bx[0] - lx)
bottomBoundary = whereZero(x[1])
topboundary = whereZero(x[1] - ly)
leftDirichlet = whereZero(x[0])
rightDirichlet = whereZero(x[0] - lx)
midBottomPoint = whereZero(x[1]) * whereZero(x[0] - .5 * lx)

# Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom
# Dbc = whereZero(x[1]) * [0, 1] + whereZero(x[1] - ly) * [0, 1] + whereZero(x[1]) * whereZero(x[0] - .5 * lx) * [1, 1]  # bind the mind point in order not to slide in x direction
# Dirichlet BC values
# Vbc = whereZero(x[1]) * [0, 0] + whereZero(x[1] - ly) * [0, vel] + whereZero(x[1]) * whereZero(x[0] - .5 * lx) * [0, 0]  # bind the mind point in order not to slide in x direction
# Neumann BC, constant confining pressure
# Nbc = whereZero(bx[0]) * [-confining, 0] + whereZero(bx[0] - lx) * [confining,
#                                                                     0]  # Neuman BC on the interplotion points
x_check = x.toListOfTuples()
bx_check = bx.toListOfTuples()

yLength = ly
# cylic load path
# vel_list = [-0.00001]*100+[0.00001]*50+\
#           [-0.00001]*160+[0.00001]*80+\
#           [-0.00001]*100+[0.00001]*50+\
#           [-0.00001]*200+[0.00001]*50+\
#           [-0.00001]*700
loadStep = 100
vel_list = [-0.0001] * loadStep
confiningList = [confining] * loadStep
if 'gaussianConfinedPressure' in loadingPath:
    confiningList = pd.read_csv('./GaussianProcess/ConfingPressureGP_10001.csv', sep=' ', header=None)
    confiningList = confiningList.values[:, numRandom]
time_start = time.time()
while t < len(vel_list):  # apply 100 load steps
    print('\n\n' + '-' * 80 + '\n' + 'Loading step # %d/%d' % (t+1, len(vel_list)))
    # the boundary condiction will not change along with the coordinate cause it was defined 
    # as [[0, 0], [0, 1], ..., [0, 0]] format escript.Data
    # before solving the PDE,  
    # strain_0 = prob.getCurrentStrain()
    # set the value of the displacement
    if 'biaxial' in loadingPath:
        confining = confiningList[t]
        Dbc = bottomBoundary * [0, 1] + topboundary * [0, 1] + midBottomPoint * [1, 1]
        Vbc = bottomBoundary * [0, 0] + topboundary * [0, vel_list[t]] + midBottomPoint * [0, 0]
        Nbc = leftBoundary * [-confining, 0] + rightBoundary * [confining, 0]
        prob.initialize(f=Nbc, specified_u_mask=Dbc, specified_u_val=Vbc)  # initialize BC
    elif 'confinedCompression' in loadingPath:
        Dbc = bottomBoundary * [0, 1] + topboundary * [0, 1] + midBottomPoint * [1, 1] + \
              leftDirichlet * [1, 0] + rightDirichlet * [1, 0]
        Vbc = bottomBoundary * [0, 0] + topboundary * [0, vel_list[t]] + midBottomPoint * [0, 0] + \
              leftDirichlet * [0, 0] + rightDirichlet * [0, 0]
        prob.initialize(specified_u_mask=Dbc, specified_u_val=Vbc)  # initialize BC
    elif 'gaussianConfinedPressure' in loadingPath:
        confining = confiningList[t]
        print('Confining pressure: %.4f MPa' % (confining / 1e6))
        Dbc = bottomBoundary * [0, 1] + topboundary * [0, 1] + midBottomPoint * [1, 1]
        Vbc = bottomBoundary * [0, 0] + topboundary * [0, vel_list[t]] + midBottomPoint * [0, 0]
        Nbc = leftBoundary * [-confining, 0] + rightBoundary * [confining, 0]
        prob.initialize(f=Nbc, specified_u_mask=Dbc, specified_u_val=Vbc)  # initialize BC
    else:
        raise ValueError('No this kind of loading path, please check the loading path')

    # l
    yLength += vel_list[t]

    t += 1
    du = prob.solve(iter_max=100, t=t)  # get solution: nod\n\nal displacement
    disp += du

    stress = prob.getCurrentStress()

    dom = prob.getDomain()  # domain is updated Lagrangian formulation
    proj = Projector(dom)
    # project the stress to the reduced domain & interpolate the stress to the boundary domain
    sig = proj(stress)
    sig_bounda = interpolate(sig, FunctionOnBoundary(dom))
    traction = matrix_mult(sig_bounda, dom.getNormal())
    tractTop = traction * topSurf
    forceTop = integrate(tractTop, where=FunctionOnBoundary(dom))
    lengthTop = integrate(topSurf, where=FunctionOnBoundary(dom))


    strain_increment = prob.getStrainIncrement()
    stress_increment = prob.getStressIncrement()
    strain = prob.getCurrentStrain()
    tangent = prob.getCurrentTangent()
    volume_strain = trace(strain)
    total_volume_strain = prob.getVolume()
    dev_strain = symmetric(strain) - volume_strain * k / dim
    shear = sqrt(2 * inner(dev_strain, dev_strain))
    strain_abs = prob.getStrainAbs()
    # save the macro information
    fout = open(os.path.join(loadInfor, 'biaxial_surf.dat'), 'a')
    fout.write(
        str(yLength / ly - 1.0) + ' ' + str(forceTop[1]) + ' ' + str(lengthTop) + ' ' + str(total_volume_strain) + '\n')
    fout.close()
    # save the the information to file loadInformation/results
    if useml:
        saveGauss2D(name=os.path.join(loadInfor, 'gauss/time_' + str(t) + '.dat'),
                    strain_increment=strain_increment,
                    strain_toatal=strain,
                    stress_increment=stress_increment,
                    stress_toatal=stress,
                    tangent=tangent)
        saveVTK(os.path.join(loadInfor, "vtk/biaxialSmooth_%d.vtu" % t),
                disp=disp, shear=shear, strain=strain, stress=stress)
    else:
        vR = prob.getLocalVoidRatio()
        fabric = prob.getLocalFabric()
        saveGauss2D(name=os.path.join(loadInfor, 'iteration_gauss/time_' + str(t) + '.dat'),
                    strain_increment=strain_increment,
                    strain_toatal=strain,  # the renewed total strain
                    stress_increment=stress_increment,
                    stress_toatal=stress,
                    tangent=tangent,
                    fabric=fabric, vR=vR,
                    strain_abs=strain_abs)
        saveVTK(os.path.join(loadInfor, "vtk/biaxialSmooth_%d.vtu" % t),
                disp=disp, shear=shear, e=vR, strain=strain, stress=stress, fabric=fabric)

# output the DEM samples as yade.gz files at the end of the simulation
# if useml == False:
    # prob.getCurrentPacking(pos=(), time=t, prefix=os.path.join(loadInfor, 'result/packing/'))
time_elapse = time.time() - time_start
fout = open(os.path.join(loadInfor, 'biaxial_surf.dat'), 'a')
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()
prob.exitSimulation()
