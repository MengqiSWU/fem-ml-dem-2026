""" Author: Ning Guo <ceguo@connect.ust.hk>
    run `mv retainingSmooth.yade.gz 0.yade.gz`
    to generate initial RVE packing

How to run this script:
    sudo apt install python-escript
    cd examples/FEMxDEM
    export PYTHONPATH="/usr/lib/python-escript:../../py/FEMxDEM"
    export LD_LIBRARY_PATH=/usr/lib/python-escript/lib
    ln -s /path/to/yade ../../py/FEMxDEM/yadeimport.py
    /path/to/yade ./retainingSmooth.py
Please amend these instructions if you find that they do not work.
"""

import tensorflow.compat.v1 as tf
from builtins import range
from esys.escript import whereZero, whereNegative, whereNonNegative, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner
from esys.finley import Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
from FEMxDEM.msFEM2D import MultiScale
from utilSelf.saveGauss import saveGauss2D
from utilSelf.general import mkdirsSelf
import time
import os
import errno
from FEMxML.netTorchDouble import Net


useml = True
mode = 'net'
vel = -0.0001
surcharge = -1e5  # surcharge equals to the initial vertical stress of the RVE packing vel<0 passive failure vel>0 active failure
B = 0.4
H = 0.2
wallH = 0.17
baseH = H - wallH  # setup dimensions
nx = 20
ny = 10  # discretization with 40x20 quads
mydomain = Rectangle(l0=B, l1=H, n0=nx, n1=ny, order=2, integrationOrder=2)
dim = mydomain.getDim()
k = kronecker(mydomain)
numg = 4 * nx * ny
nump = 6
packNo = list(range(0, numg, 16))
rtol=1e-2


loadInfor = ('../simu/ML_retaining_%s_OnlyInclude_Nodouble' % mode if useml else '../simu/DEM_retaining') + '_' + str(nx) + '_' + str(ny)

mkdirsSelf(
    loadInfor,
    os.path.join(loadInfor, 'gauss'),
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'packing'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_vtk'),
    os.path.join(loadInfor, 'iteration_packing'))

print('-' * 80)
print('\tML mode: %s' % mode if useml else '\tDEM-FEM')
print('B:\t%.5f' % B + '\t nx:\t%d' % nx)
print('H:\t%.5f' % H + '\t ny:\t%d' % ny)
# print('confing:\t%e' % confining)
print('loading velocity:\t%e' % vel)
print('err tolerance:\t%e' % rtol)
print('Number of processor:\t%d' % nump)
# print('Axial strain:\t%f' % axialStrain + '\t NumSubstep:\t%d' % loadSubStep)
print(loadInfor)
print('-' * 80)


disp = Vector(0., Solution(mydomain))

prob = MultiScale(domain=mydomain, ng=numg, np=nump,
                  useml=useml, mode=mode,
                  random=False, rtol=rtol, usePert=False, pert=-2.e-5, verbose=True,
                  loadInfor=loadInfor)

t = 0
time_start = time.time()

x = mydomain.getX()
bx = FunctionOnBoundary(mydomain).getX()
left = whereZero(x[0])
right = whereZero(x[0] - B)
bottom = whereZero(x[1])
top = whereZero(bx[1] - H)
base = whereZero(x[0] - B) * whereNegative(x[1] - baseH)
wall = whereZero(x[0] - B) * whereNonNegative(x[1] - baseH)
wallBF = whereZero(bx[0] - B) * whereNonNegative(bx[1] - baseH)
# Dirichlet BC, all fixed in space except wall (only fixed in x direction, smooth)
Dbc = left * [1, 1] + base * [1, 1] + bottom * [1, 1] + wall * [1, 0]
Vbc = left * [0, 0] + base * [0, 0] + bottom * [0, 0] + wall * [vel, 0]
# Neumann BC, apply surcharge at the top surface
Nbc = top * [0, surcharge]

stress = prob.getCurrentStress()
proj = Projector(mydomain)
sig = proj(stress)
sig_bounda = interpolate(sig, FunctionOnBoundary(mydomain))
traction = matrix_mult(sig_bounda, mydomain.getNormal())
tract = traction * wallBF  # traction on wall
forceWall = integrate(tract, where=FunctionOnBoundary(mydomain))  # force on wall
lengthWall = integrate(wallBF, where=FunctionOnBoundary(mydomain))
# open('./result/pressure.dat', 'w')
fout = open(os.path.join(loadInfor, 'pressure.dat'), 'w')
fout.write('0 ' + str(forceWall[0]) + ' ' + str(lengthWall) + ' '+str(1.) + '\n')
fout.close()

while t < 100:
    print('\n\n' + '-' * 80 + '\n' + 'Loading step # %d/%d' % (t, 100))
    prob.initialize(f=Nbc, specified_u_mask=Dbc, specified_u_val=Vbc)
    t += 1
    du = prob.solve(iter_max=100, t=t)

    disp += du
    stress = prob.getCurrentStress()

    dom = prob.getDomain()
    proj = Projector(dom)
    sig = proj(stress)

    sig_bounda = interpolate(sig, FunctionOnBoundary(dom))
    traction = matrix_mult(sig_bounda, dom.getNormal())
    tract = traction * wallBF
    forceWall = integrate(tract, where=FunctionOnBoundary(dom))
    lengthWall = integrate(wallBF, where=FunctionOnBoundary(dom))
    total_volume_strain = prob.getVolume()
    fout = open(os.path.join(loadInfor, 'pressure.dat'), 'a')
    fout.write(str(t * vel) + ' ' + str(forceWall[0]) + ' ' + str(lengthWall) + ' ' + str(total_volume_strain)+'\n')
    fout.close()

    vR = prob.getLocalVoidRatio()
    rotation = prob.getLocalAvgRotation()
    fabric = prob.getLocalFabric()
    strain = prob.getCurrentStrain()
    saveGauss2D(
        # name='./result/gauss/time_' + str(t) + '.dat',
        name=os.path.join(loadInfor, 'gauss/time_' + str(t) + '.dat'),
        strain=strain, stress=stress, fabric=fabric)
    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * k / dim
    shear = sqrt(2 * inner(dev_strain, dev_strain))
    saveVTK(
        # "./result/vtk/retainingSmooth_%d.vtu" % t,
        os.path.join(loadInfor, "vtk/retainingSmooth_%d.vtu" % t),
            disp=disp, stress=stress, shear=shear, e=vR, rot=rotation)

prob.getCurrentPacking(pos=packNo, time=t, prefix=os.path.join(loadInfor, 'packing/'))
time_elapse = time.time() - time_start
fout = open(os.path.join(loadInfor, 'pressure.dat'), 'a')
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()
prob.exitSimulation()
