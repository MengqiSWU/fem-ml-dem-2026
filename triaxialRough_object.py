import copy
import os
import time
from builtins import range
import numpy as np
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, Tensor, \
    Function, wherePositive, whereNegative, inf, sup
from esys.escript.pdetools import Projector
from esys.finley import Brick, ReadGmsh
from esys.weipa import saveVTK

from utilSelf.esysUtils import boudaryCondition_3d, force_length_calculation
from utilSelf.general import check_mkdir, get_pool, writeLine, echo
from utilSelf.saveGauss import saveGauss3D
from utilSelf.general import getCons
from FEMxDEM.implicitSolver import ImplicitSolver

# ---------------- parameters ------------------------------
mode = 'csuh'  # 'mcc' 'net' 'dem' 'csuh' 'mises' 'lade' 'uh' 'norsand'
ocr = 100.0
e0 = 0.52
confining = p0 = 1.e6  # confining pressure
dim = 3
num_mesh = 9
nx, ny, nz = num_mesh, num_mesh, num_mesh * 2
threads = 12
pool = get_pool(mpi=False, threads=threads)
loadingPath = 'conventionalDisp'  # TriaxialRough consolidation conventionalP conventionalDisp undrained

# ----------------------------------------------------------------
lx = 0.05
ly = 0.05
order = 2
lz = 0.1  # sample dimension
# loading information
'''
    In CSUH model, the loading displacement increment will matter in 
    the return mapping calculation
'''
loadStep_vel = 200
axialStrain = 0.15
vel = axialStrain * lz / loadStep_vel
vel_list = [vel] * loadStep_vel

out_directory = '../simu/%s' % mode
check_mkdir(out_directory)
loadInfor = '../simu/%s/%s_%s_%d_%d_%d_3D' % (mode, mode, loadingPath, nx, ny, nz)
if 'csuh' in mode or 'uh' in mode:
    loadInfor += '_ocr%.3f_p%dkPa' % (ocr, p0/1e3)
if 'norsand' in mode:
    loadInfor += '_e0%.3f_p%dkPa' % (e0, p0/1e3)

check_mkdir(
    loadInfor,
    os.path.join(loadInfor, 'gauss'),
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'packing'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_vtk'),
    os.path.join(loadInfor, 'iteration_packing'))

if num_mesh == 77:
    mydomain = ReadGmsh('cylinder1.msh', numDim=dim, order=order, integrationOrder=2)
    numg = len(Tensor(0, Function(mydomain)).toListOfTuples())
else:
    mydomain = Brick(l0=lx, l1=ly, l2=lz, n0=nx, n1=ny, n2=nz, order=order,
                     integrationOrder=2)  # 20-noded,8-Gauss hexahedral element
    numg = len(Function(mydomain).getX().toListOfTuples())

echo(loadInfor,
     'Number of Gaussian points: #%d' % numg)

kwargs = {'p0': p0, 'ocr': ocr, 'chi':0.0, 'kappa':0.01, 'e0':e0}
cons = getCons(mode=mode, numg=numg, explicitFlag=False, **kwargs, pool=pool, ndim=mydomain.getDim())

prob = ImplicitSolver(domain=mydomain, cons=cons, pool=pool, loadInfor=loadInfor,
                      save_loading_flag=False)  # mpi is activated

x = mydomain.getX()
bx = FunctionOnBoundary(mydomain).getX()
nx_, nx = whereZero(x[0]-inf(x[0])), whereZero(x[0] - sup(x[0]))
ny_, ny = whereZero(x[1]-inf(x[1])), whereZero(x[1] - sup(x[1]))
nz_, nz = whereZero(x[2]-inf(x[2])), whereZero(x[2] - sup(x[2]))
where_confining = wherePositive(bx[2]-inf(bx[2]))*whereNegative(bx[2]-sup(bx[2]))
fz = whereZero(bx[2] - sup(bx[2]))
fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))
fy_, fy = whereZero(bx[1] - inf(bx[1])), whereZero(bx[1] - sup(bx[1]))
forceTop, areaTop = force_length_calculation(sig=prob.sig, domain=mydomain, where=fz)

fname = os.path.join(loadInfor, 'biaxial_surf.dat')
writeLine(fname=fname, mode='w', s='AxialStrain forceTop lengthTop volumeStrain\n')
writeLine(fname=fname, mode='a', s='%.3e %.3e %.3e %.3e\n' % (0., forceTop[2], areaTop, 0.))

t = 0
time_start = time.time()
loadingStep = len(vel_list)
zLength = copy.deepcopy(lz)
while t < loadingStep:
    echo('%s Time %d/%d %s' % (' ' * 25, t, loadingStep, ' '*20))
    vel = vel_list[t]
    # vel_remain = copy.deepcopy(vel)
    # scaler, remain_du = 1.0, 1.0
    # du = Vector(0., Solution(mydomain))
    q, r, y = boudaryCondition_3d( nx_=nx_, nx=nx, ny_=ny_, ny=ny,
        nz_=nz_, nz=nz, fx_=fx_, fx=fx, fy_=fy_, fy=fy,  where_confining=where_confining, vel=vel,
        out_normal=prob.domain.getNormal(), confining=confining)

    prob.initialize(y=y, q=q, r=r)
    zLength -= vel
    prob.solve(iter_max=20, t=t)

    #     if converge == False:
    #         scaler = 0.5 * scaler
    #         if scaler < 1. / 2 ** 4:
    #             raise ValueError('=' * 80 + '\n \t Can not converge after 4 times split.' +
    #                              '\n \t please decrease the loading step and retry.')
    #         vel = 0.5 * vel
    #         print('\n\t\t Can not converge, original vel_0: %.3e vel_1: %.3e' % (2. * vel, vel))
    #     else:
    #         remain_du = remain_du - scaler
    #         du += ddu
    #     if remain_du <= 0.:
    #         break
    # t += 1

    disp = prob.u
    stress = prob.sig

    forceTop, areaTop = force_length_calculation(sig=stress, domain=mydomain, where=fz)

    strain = prob.eps
    stress = prob.sig
    volume_strain = trace(strain)
    saveVTK(os.path.join(loadInfor, "vtk/triaxialRough_%d.vtu" % t), disp=disp, strain=strain, stress=stress,
            voleps=volume_strain)
    total_volume_strain = np.average(np.array(volume_strain.toListOfTuples()))
    s = '%.3e %.3e %.3e %.3e\n' % (zLength / lz - 1.0, forceTop[2], areaTop, total_volume_strain)
    writeLine(fname=fname, s=s, mode='a')
    t += 1


time_elapse = time.time() - time_start
writeLine(fname, "#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n', 'a')
if pool:
    pool.close()
