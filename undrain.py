""" Author: Ning Guo <ceguo@connect.ust.hk>
    run `mv undrain.yade.gz 0.yade.gz`
    to generate initial RVE packing

How to run this script:
    sudo apt install python-escript
    cd examples/FEMxDEM
    export PYTHONPATH="/usr/lib/python-escript:../../py/FEMxDEM"
    export LD_LIBRARY_PATH=/usr/lib/python-escript/lib
    ln -s /path/to/yade ../../py/FEMxDEM/yadeimport.py
    /path/to/yade ./undrain.py
Please amend these instructions if you find that they do not work.
"""

from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution
from esys.finley import Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
from FEMxDEM.msFEMup import MultiScale
from utilSelf.saveGauss import saveGauss2D
from utilSelf.general import mkdirsSelf
from utilSelf.girdPlot import gridPlot
import time
import os
import errno
from utilSelf.general import mkdirsSelf
import sys


def get_pool(mpi=False, threads=1):
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

# mpi_pool = get_pool(mpi=True)
loadingPath = 'Undrained'
mode = 'find'
useml = False
nx = 2
ny = 4  # discretization
loadInfor = ('../simu/ML_%s' % mode if useml else '../simu/Test_ABS_DEM') + '_' + str(nx) + '_' + str(
    ny) + '_' + loadingPath

print('\n'+'-'*80)
print(loadInfor)

# mkdirs
mkdirsSelf(
    loadInfor,
    os.path.join(loadInfor, 'gauss'),
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'packing'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_vtk'),
    os.path.join(loadInfor, 'iteration_packing'))

confining = -2.e5
pore = 1.e5  # initial pore pressure
perm = 0.001 ** 2 / (180. * 8.9e-4)  # unscaled permeability, using KC equation
kf = 2.2e9  # fluid bulk modulus
dt = .1
vel = -0.0001  # time step and loading speed
lx = 0.05
ly = 0.1  # sample dimension
mydomain = Rectangle(l0=lx, l1=ly, n0=nx, n1=ny, order=2, integrationOrder=2)
k = kronecker(mydomain)
dim = 2.
numg = 4 * nx * ny  # no. of Gauss points
mpi = False  # use MPI

prob = MultiScale(domain=mydomain, pore0=pore, perm=perm, kf=kf, dt=dt, ng=numg, useMPI=False, rtol=1.e-2, np=6, mpipool=None)
disp = Vector(0., Solution(mydomain))
t = 0
ux = mydomain.getX()  # disp. node coordinate
px = ReducedSolution(mydomain).getX()  # press. node coordinate
bx = FunctionOnBoundary(mydomain).getX()
topSurf = whereZero(bx[1] - ly)
uDbc = whereZero(ux[1]) * [0, 1] + whereZero(ux[0] - lx / 2.) * whereZero(ux[1]) * [1, 1] + whereZero(ux[1] - ly) * [0,
                                                                                                                     1]  # disp. Dirichlet BC mask
vDbc = whereZero(ux[1]) * [0, 0] + whereZero(ux[0] - lx / 2.) * whereZero(ux[1]) * [0, 0] + whereZero(ux[1] - ly) * [0,
                                                                                                                     vel * dt]  # disp. Dirichlet BC value
uNbc = whereZero(bx[0]) * [-confining, 0] + whereZero(bx[0] - lx) * [confining, 0]  # disp. Neumann BC

stress = prob.getCurrentStress()  # effective stress at GP
proj = Projector(mydomain)
sig = proj(stress)  # effective stress at node (reduced)
sig_bound = interpolate(sig, FunctionOnBoundary(mydomain))
traction = matrix_mult(sig_bound, mydomain.getNormal())
tractTop = traction * topSurf
forceTop = integrate(tractTop, where=FunctionOnBoundary(mydomain))
lengthTop = integrate(topSurf, where=FunctionOnBoundary(mydomain))
fout = open(os.path.join(loadInfor, 'resultantForce.dat'), 'w')
fout.write('0 ' + str(forceTop[1]) + ' ' + str(lengthTop) + '\n')

time_start = time.time()
while t < 400:
    print('\n'+'-'*20+'TIME %d' % t + '-'*20)
    prob.initialize(f=uNbc, umsk=uDbc, uvalue=vDbc)
    t += 1
    du = prob.solve(globalIter=10, solidIter=50)
    disp += du
    pore = prob.getCurrentPore()  # pore pressure at node (reduced)
    flux = prob.getCurrentFlux()  # Darcy flux at GP
    stress = prob.getCurrentStress()  # effective stress at GP
    strain = prob.getCurrentStrain()  # disp. grad at GP
    volume_strain = trace(strain)  # volumetric strain
    dev_strain = symmetric(strain) - volume_strain * k / dim  # deviatoric strain
    shear = sqrt(2. * inner(dev_strain, dev_strain))  # shear strain
    fab = prob.getLocalFabric()  # fabric tensor at GP
    dev_fab = 4. * (fab - trace(fab) / dim * k)
    anis = sqrt(.5 * inner(dev_fab, dev_fab))
    p = prob.getEquivalentPorosity()  # porosity at GP
    rot = prob.getLocalAvgRotation()  # average rotation at GP
    saveGauss2D(name=os.path.join(loadInfor, 'gauss/time_' + str(t) + '.dat'), strain=strain, fabric=fab, stress=stress)
    dom = prob.getDomain()  # domain updated (Lagrangian)
    proj = Projector(dom)
    flux = proj(flux)  # Darcy flux at node (reduced)
    p = proj(p)  # porosity at node (reduced)
    shear = proj(shear)  # shear strain at node (reduced)
    anis = proj(anis)
    rot = proj(rot)
    saveVTK(os.path.join(loadInfor, 'vtk/undrain_%d.vtu' % t), disp=disp, pore=pore, flux=flux, shear=shear, p=p,
            anis=anis, rot=rot)
    sig = proj(stress)  # effective stress at node (reduced)
    sig_bound = interpolate(sig, FunctionOnBoundary(dom))
    traction = matrix_mult(sig_bound, dom.getNormal())
    tractTop = traction * topSurf
    forceTop = integrate(tractTop, where=FunctionOnBoundary(dom))
    lengthTop = integrate(topSurf, where=FunctionOnBoundary(dom))
    fout.write(str(t * vel * dt / ly) + ' ' + str(forceTop[1]) + ' ' + str(lengthTop) + '\n')

prob.getCurrentPacking(time=t, prefix=os.path.join(loadInfor, 'packing'))
time_elapse = time.time() - time_start
fout.write('#Elapsed time in hours: ' + str(time_elapse / 3600.) + '\n')
fout.close()
prob.exitSimulation()
