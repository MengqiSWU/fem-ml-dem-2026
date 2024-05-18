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

from utilSelf.esysUtils import boudaryCondition_3d_PBC, force_length_calculation, get_veps_seps
from utilSelf.general import check_mkdir, get_pool, writeLine, echo
from utilSelf.general import check_mkdir, get_pool, writeLine, getCons, echo, get_load_information,\
    explicit_material_constants

from utilSelf.saveGauss import saveGauss3D
from utilSelf.general import getCons

# from FEMxDEM.implicitSolver import ImplicitSolver
from FEMxDEM.implicitSolver_3d_accum import ImplicitSolver
# from FEMxDEM.implicitSolver_3d import ImplicitSolver


# ---------------- geometry parameters ------------------------------
lx = 0.5
ly = 0.5
lz = 1.0
dim = 3   # sample dimension
num_mesh = 5
nx, ny, nz = num_mesh, num_mesh, num_mesh * 2
order = 1
mydomain = Brick(l0=lx, l1=ly, l2=lz, n0=nx, n1=ny, n2=nz, order=order,
                     integrationOrder=2)  # 20-noded,8-Gauss hexahedral element
numg = len(Function(mydomain).getX().toListOfTuples())
nump = 12 # number of processes in multiprocessing



# ---------------- loading parameters ------------------------------
confining = p0 = 100e3  # confining pressure
# loadStep_vel = 100
# axialStrain = 0.05
# vel = axialStrain * lz / loadStep_vel
# vel_list = [vel] * loadStep_vel

#
# vel = 0.001
# vel_list = [vel] * 80  # Monotonouw



vel = 0.001
# vel_ld1= [vel * 0.5] * 20
# vel_unld= [-vel * 0.5] * 12
# vel_reld1 = [vel* 0.5] * 12
vel_ld1= [vel] * 10
vel_unld= [-vel] * 6
vel_reld1 = [vel] * 6

vel_reld= [vel] * 35
vel_list = vel_ld1 + vel_unld + vel_reld1 + vel_reld   # one circle









rtol = 1e-2

# ---------------------------Cons model-------------------------------------
mode = 'dem3d'  # 'mcc' 'net' 'dem' 'dem3d' 'csuh' 'mises' 'lade' 'uh' 'norsand'
active_iter = None

NN_sig_path = 'X_epsAND3d_Y_sig_dddd20_Fourier_noRotate_FEM_DEM_sig'
NN_D_path = 'X_epsAND3d_Y_D_dddd20_Fourier_noRotate_FEM_DEM_D'




p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        # p0=confining,
        nn_name=None,
        # nn_name = NN_sig_path,
        # nn_name_D = NN_D_path,
        # active_iter = active_iter,
        # csuh_para_line=None,
    )


# for split_D
# p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
#     explicit_material_constants(
#         # p0=confining,
#         # nn_name=None,
#         nn_name=NN_sig_path,
#         nn_name_Dv=NN_Dv_path,
#         nn_name_Dr=NN_Dr_path,
#         # active_iter = active_iter,
#         # csuh_para_line=None,
#     )



# for split_sig
# p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
#     explicit_material_constants(
#         p0=confining,
#         # nn_name=None,
#         nn_name_sigv=NN_sigv_path,
#         nn_name_sigr=NN_sigr_path,
#         nn_name_D=NN_D_path,
#         # active_iter = active_iter,
#         # csuh_para_line=None,
#     )



# ---------------------------Save path-------------------------------------
seMPI = False
explicitFlag = False
# loadingPath = 'pressureBiax'  # 'biaxial' 'confinedCompression' 'gaussianConfinedPressure' 'pressureBiax'
smoothFlag = False

out_directory = '../simu/Triaxial_PBC'
check_mkdir(out_directory)
loadInfor = get_load_information(
    out_directory=out_directory, test_name='biax_3d', mode=mode, smooth_flag=smoothFlag, explicit_flag=explicitFlag,
      nx=nx, ny=ny, order=order, numg=numg, **kwargs)


loadInfor += '_Y3e8_fri0.5_p0.3_rM01_n1000_nodenser_accum_rotation'
kwargs['save_path'] = loadInfor



check_mkdir(
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_packing')
)



# ---------------------------Solving-------------------------------------
# cons = getCons(mode=mode, numg=numg, nump=nump, explicitFlag=explicitFlag, **kwargs, ndim=mydomain.getDim()) #拿到初始scene[sig, D] from RVE
cons = getCons(mode=mode, numg=numg, pool=None, nump=nump, explicitFlag=explicitFlag, **kwargs)



prob = ImplicitSolver(domain=mydomain, cons=cons,  loadInfor=loadInfor,
                      save_loading_flag=True if mode == 'dem3d' or mode == 'dem' else False)  # mpi is activated

echo(
    'CWD:           %s' % loadInfor,
    'Mode:          %s' % mode,
    'Num Porcess:   %d' % nump,
    'Mesh infor:    nx_%d ny_%d nz_%d' % (nx, ny, nz),
    'Order:         %d' % order,
    'Num Guass:     %d' % numg,
    'Confining:     %d Pa' % confining,
    'Load_vel:      %.3e m/s' % vel,
    )




x = mydomain.getX()   #find node coordinate
bx = FunctionOnBoundary(mydomain).getX()  #find  coordinates of gaussian points
nx_, nx = whereZero(x[0]-inf(x[0])), whereZero(x[0] - sup(x[0]))  #find planes where x=0 and x=0.5
ny_, ny = whereZero(x[1]-inf(x[1])), whereZero(x[1] - sup(x[1]))  #find planes where y=0 and y=0.5
nz_, nz = whereZero(x[2]-inf(x[2])), whereZero(x[2] - sup(x[2]))  #find planes where z=0 and z=1.0
where_confining = wherePositive(bx[2]-inf(bx[2]))*whereNegative(bx[2]-sup(bx[2])) #exclude Gaussian points on top and bottom planes
fz = whereZero(bx[2] - sup(bx[2]))  #apply force to planes where z=1.0
fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))  #find gaussian points where x=0 and x=0.5
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
    print('\n\n' + '-' * 80 + '\n' + 'Loading step # %d/%d time: %.2e mins' % (
    t, len(vel_list), (time.time() - time_start) / 60.))
    vel = vel_list[t]

    q, r, y = boudaryCondition_3d_PBC( nx_=nx_, nx=nx, ny_=ny_, ny=ny,
        nz_=nz_, nz=nz, fx_=fx_, fx=fx, fy_=fy_, fy=fy,  where_confining=where_confining, vel=vel,
        out_normal=prob.domain.getNormal(), confining=confining)
    prob.initialize(y=y, q=q, r=r)
    zLength -= vel
    prob.solve(iter_max=20, t=t)  #,eps_last= eps_last

    disp = prob.u
    stress = prob.sig

    forceTop, areaTop = force_length_calculation(sig=stress, domain=mydomain, where=fz)
    veps, seps = get_veps_seps(strain=prob.eps, domain=prob.domain)
    strain = prob.eps
    stress = prob.sig
    volume_strain = trace(strain)


    vtkFileName = os.path.join(loadInfor, 'vtk/biaxial_%s_%s_%d.vtu' %
                               ('smooth' if smoothFlag else 'rough', mode, t))
    saveVTK(vtkFileName, disp=disp, stress=stress, shear=seps, vol_eps=veps)

    total_volume_strain = np.average(np.array(volume_strain.toListOfTuples()))
    s = '%.3e %.3e %.3e %.3e\n' % (zLength / lz - 1.0, forceTop[2], areaTop, total_volume_strain)
    writeLine(fname=fname, s=s, mode='a')
    t += 1


time_elapse = time.time() - time_start
fout = open(os.path.join(loadInfor, 'biaxial_surf.dat'), 'a')
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()

