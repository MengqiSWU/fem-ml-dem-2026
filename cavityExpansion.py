from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution, Data, whereNonPositive, wherePositive, inf, Tensor, \
    Function,Scalar
from esys.finley import ReadGmsh, Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
import time
import os, sys
import numpy as np
import errno
from utilSelf.general import check_mkdir, get_pool, writeLine, getCons, echo, get_load_information,\
    explicit_material_constants
from utilSelf.esysUtils import boudaryConditions2D, force_length_calculation, get_veps_seps, boudaryCondition_cavityexpansion
from FEMxDEM.implicitSolver import ImplicitSolver
from FEMxDEM.explicitSolver import explicitSolver

useMPI = False
explicitFlag = False
seMPI = False
confining = p0 = 100e3  # confining pressure
order = 1
nump = 12

vel = 0.1
vel_list = [vel] * 70





# --------------------------Mesh-------------------------------

rc, rout = 15, 150
mesh_name = 'Cavity_quarter'
mydomain = ReadGmsh('./meshes/cavity_msh/%s.msh' % mesh_name, numDim=2,
                    order=order, integrationOrder=2)
# print([m for m in dir(mydomain) if "Tag" in m])
# print([m for m in dir(FunctionOnBoundary(mydomain)) if "Tag" in m])
rtol = 1e-2
numg = len(Vector(0, Function(mydomain)).toListOfTuples())






# --------------------------Model-------------------------------
mode = 'csuh'  # 'ml' 'dem' 'elastic' 'vonmises' 'vonmisesml' 'vonmisessemi' 'mcc' 'mldem'
active_iter = None
NN_sig_path = 'X_epsAND3f_Y_sig_ddd14_Fourier_noRotate_FEM_DEM_sig'
NN_D_path = 'X_epsAND3f_Y_D_ddd14_Fourier_noRotate_FEM_DEM_D'




# material properties #remember to change the path in different simulation cases
p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        # p0=confining,
        # nn_name=NN_sig_path,
        # nn_name_D=NN_D_path,
        # active_iter = active_iter,
        nn_name=None,
        # csuh_para_line=None,
    )







# ---------------------------Save path-------------------------------------

out_directory = '../simu/cavity'
check_mkdir(out_directory)
loadInfor = get_load_information(
    out_directory=out_directory, test_name='cavity_quater', mode=mode,  explicit_flag=explicitFlag,
    order=order, numg=numg, mesh_name=mesh_name, **kwargs)
kwargs['save_path'] = loadInfor

check_mkdir(
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_packing')
)


# ---------------------------mask point-------------------------------------

x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()
FSb = FunctionOnBoundary(mydomain)
FSx = x.getFunctionSpace()

nx  = whereZero(x[0]-inf(x[0]))  #mask左侧边界点
ny  = whereZero(x[1]-inf(x[1]))     #mask下侧边界点
fx  = whereZero(bx[0] - inf(bx[0]))  #mask左侧边界内的高斯点
fy  = whereZero(bx[1] - inf(bx[1]))  #mask下侧边界内的高斯点 （被mask的地方值为1）

tag_inner = mydomain.getTag("inner_boundary")  # 1
tag_outer = mydomain.getTag("outer_boundary")  # 2

n_in  = Scalar(0.0, FSx);n_in.setTaggedValue(tag_inner, 1.0)  #mask内圆边界点
n_out = Scalar(0., FSx); n_out.setTaggedValue(tag_outer, 1.0) #mask外圆边界点

f_in  = Scalar(0., FSb); f_in.setTaggedValue(tag_inner, 1.0)    #mask内圆边界内高斯点
f_out = Scalar(0., FSb); f_out.setTaggedValue(tag_outer, 1.0) #mask外圆边界内高斯点



# ---------------------------Solving-------------------------------------

cons = getCons(mode=mode, numg=numg, pool=None, nump=nump, explicitFlag=explicitFlag, **kwargs)

prob = ImplicitSolver(domain=mydomain, cons=cons,  loadInfor=loadInfor,
                      save_loading_flag=True if mode == 'csuh' or mode == 'dem' else False)  # mpi is activated

echo(
    'CWD:           %s' % loadInfor,
    'Mode:          %s' % mode,
    'Num Porcess:   %d' % nump,
    'Mesh infor:    %s' % mesh_name,
    'Order:         %d' % order,
    'Num Guass:     %d' % numg,
    'Confining:     %d Pa' % confining,
    'Load_vel:      %.3e m/s' % vel,
    )



t = 0
time_start = time.time()
loadingStep = len(vel_list)
# zLength = copy.deepcopy(lz)




while t < loadingStep:
    print('\n\n' + '-' * 80 + '\n' + 'Loading step # %d/%d time: %.2e mins' % (
    t, len(vel_list), (time.time() - time_start) / 60.))
    vel = vel_list[t]

    q, r, y = boudaryCondition_cavityexpansion(nx=nx, ny=ny, n_in=n_in, n_out=n_out, fx=fx, fy=fy, f_in=f_in, f_out=f_out,
                                               rc=rc, vel = vel, domain=prob.domain, confining=confining,
                                               out_normal=prob.domain.getNormal())
    prob.initialize(y=y, q=q, r=r)
    # zLength -= vel
    prob.solve(iter_max=20, t=t)  #,eps_last= eps_last

    disp = prob.u
    stress = prob.sig

    # forceTop, areaTop = force_length_calculation(sig=stress, domain=mydomain, where=fz)
    veps, seps = get_veps_seps(strain=prob.eps, domain=prob.domain)
    strain = prob.eps
    stress = prob.sig
    volume_strain = trace(strain)


    vtkFileName = os.path.join(loadInfor, 'vtk/cavity_%s_%d.vtu' % (mode, t))
    saveVTK(vtkFileName, disp=disp, stress=stress, shear=seps, vol_eps=veps)

    total_volume_strain = np.average(np.array(volume_strain.toListOfTuples()))
    # s = '%.3e %.3e %.3e %.3e\n' % (zLength / lz - 1.0, forceTop[2], areaTop, total_volume_strain)
    # writeLine(fname=fname, s=s, mode='a')
    t += 1


time_elapse = time.time() - time_start
fout = open(os.path.join(loadInfor, 'cavity_surf.dat'), 'a')
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()


