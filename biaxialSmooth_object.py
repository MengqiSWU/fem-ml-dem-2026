from builtins import range
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution, Data, whereNonPositive, wherePositive, inf, Tensor, \
    Function
from esys.finley import ReadGmsh, Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
import time
import os, sys
import numpy as np
import errno
from utilSelf.general import check_mkdir, get_pool, writeLine, getCons, echo, get_load_information,\
    explicit_material_constants
from utilSelf.esysUtils import boudaryConditions2D, force_length_calculation, get_veps_seps, get_boundary_u_traction
from FEMxDEM.implicitSolver import ImplicitSolver
from FEMxDEM.explicitSolver import explicitSolver

useMPI = False
explicitFlag = False
# loadingPath = 'pressureBiax'  # 'biaxial' 'confinedCompression' 'gaussianConfinedPressure' 'pressureBiax'
smoothFlag = False
numRandom = 0
nump = 10  # number of processes in multiprocessing
confining = p0 = 1.e5  # confining pressure
lx = 0.5
ly = 1.0  # sample size, 50mm by 100mm
axialStrain = 0.06
loadStep = 80
#vel = axialStrain*ly/loadStep
# vel_list = [vel] * loadStep  # original loading


# # loadStep = int(abs(ly * axialStrain / vel)) + 1

# reversal loading
# vel = 0.001
# vel_ld= [vel] * 15
# vel_unld= [-vel] * 10
# vel_reld= [vel] * 55

# reversal loading2
vel = 0.001
vel_ld= [vel] * 12
vel_unld= [-vel] * 8
vel_reld= [vel] * 30

# reversal loading2
# vel = 0.001
# vel_ld= [vel] * 10
# vel_unld= [-vel] * 8
# vel_reld= [vel] * 32

# Mono loading2
# vel = 0.001
# vel_ld= [vel] * 12
# vel_unld= [-vel] * 8
# vel_reld= [vel] * 14
#
vel_list = vel_ld + vel_unld + vel_reld



mode = 'dem'  # 'ml' 'dem' 'elastic' 'vonmises' 'vonmisesml' 'vonmisessemi' 'mcc' 'mldem'
active_iter = None
NN_sig_path = 'X_epsAND3f_Y_sig_ddd14_Fourier_noRotate_FEM_DEM_sig'
NN_D_path = 'X_epsAND3f_Y_D_ddd14_Fourier_noRotate_FEM_DEM_D'



# --------------------------Mesh size----------------------------
mesh_number = 2
nx, ny = mesh_number, mesh_number * 2  # sample discretization, 8 by 16 quadrilateral elements
order = 1

# mesh_name = 'biaxial_0.1_162'
# mydomain = ReadGmsh('./meshes/biaxial_msh/%s.msh' % mesh_name, numDim=2,
#                    order=1, integrationOrder=2)

#
mydomain = Rectangle(l0=lx, l1=ly, n0=nx, n1=ny,
                     order=1, integrationOrder=2)

rtol = 1e-2
numg = len(Vector(0, Function(mydomain)).toListOfTuples())

# material properties #remember to change the path in different simulation cases
p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining,
        # nn_name=NN_sig_path,
        # nn_name_D=NN_D_path,
        # active_iter = active_iter,
        nn_name=None,
        # csuh_para_line=None,
    )


# create the simulation directory
out_directory = '../simu/biaxial_Reld'
check_mkdir(out_directory)

# remember to change the loading path
loadInfor = get_load_information(
    out_directory=out_directory, test_name='biax', mode=mode, smooth_flag=smoothFlag, explicit_flag=explicitFlag,
      nx=nx, ny=ny, order=order, numg=numg, **kwargs)
kwargs['save_path'] = loadInfor

# loadInfor= loadInfor + '_Reld_St15_modified_H'
loadInfor= loadInfor + '_Reld_St12_2H'
# loadInfor= loadInfor + '_Mono_2H'

check_mkdir(
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'iteration_packing')
)

cons = getCons(mode, numg=numg, pool=None, nump=nump, explicitFlag=explicitFlag, **kwargs)

prob = ImplicitSolver(domain=mydomain, cons=cons, loadInfor=loadInfor,
    save_loading_flag=True if mode == 'dem' or 'mldem' else False)


echo(
    'CWD:           %s' % loadInfor,
    'Mode:          %s' % mode,
    'Num Porcess:   %d' % nump,
    'Mesh infor:    nx_%d ny_%d' % (nx, ny),
    'Order:         %d' % order,
    'Num Guass:     %d' % numg,
    'Confining:     %d Pa' % confining,
    'Load_vel:      %.3e m/s' % vel,
)

# get initial boundary
nx_, nx, ny_, ny, fx_, fx, fy_, fy, mid_ny_ = \
    get_boundary_u_traction(domain=mydomain)

dim = mydomain.getDim()
k = kronecker(mydomain)
gaussianPointsCoordinate = Function(mydomain).getX()

# disp_check = disp.toListOfTuples()
t = 0
bixial_fname = os.path.join(loadInfor, 'biaxial_surf.dat')
forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=mydomain, where=fy)
writeLine(fname=bixial_fname, mode='w', s='AxialStrain forceTop lengthTop volumeStrain\n')
line = '%.3e %.3e %.3e %.3e\n' % ((ly / ly - 1.0), forceTop[1], lengthTop, 0.)
writeLine(fname=bixial_fname, mode='a', s=line)
yLength = ly

time_start = time.time()

t_total = len(vel_list)
while t < t_total:  # apply 100 load steps
    print('\n\n' + '-' * 80 + '\n' + 'Loading step # %d/%d time: %.2e mins' % (t, len(vel_list), (time.time()-time_start)/60.))
    q, r, y, Y = boudaryConditions2D(
        nx_=nx_, ny_=ny_, nx=nx, ny=ny, mid_ny_=mid_ny_,
        fx_=fx_, fy_=fy_, fx=fx, fy=fy,
        confining=confining, smoothFlag=smoothFlag, vel=vel_list[t])
    prob.initialize(Y=Y, y=y, q=q, r=r)

    prob.solve(iter_max=20, t=t)  # get solution: nod\n\nal displacement
    disp = prob.u
    stress = prob.sig
    yLength = np.sum((prob.domain.getX()[1] * ny).toListOfTuples()) / np.sum(ny.toListOfTuples())
    forceTop, lengthTop = force_length_calculation(sig=stress, domain=mydomain, where=fy)

    veps, seps = get_veps_seps(strain=prob.eps, domain=prob.domain)
    # save the macro information
    line = '%.3e %.3e %.3e %.3e\n' % ((yLength / ly - 1.0), forceTop[1], lengthTop, np.average(veps.toListOfTuples()))
    writeLine(fname=bixial_fname, mode='a', s=line)
    # save the the information to file loadInformation/results
    vtkFileName = os.path.join(loadInfor, 'vtk/biaxial_%s_%s_%d.vtu' %
                               ('smooth' if smoothFlag else 'rough', mode, t))
    saveVTK(vtkFileName, disp=disp, stress=stress, shear=seps, vol_eps=veps)
    t += 1

# output the DEM samples as yade.gz files at the end of the simulation
time_elapse = time.time() - time_start
fout = open(os.path.join(loadInfor, 'biaxial_surf.dat'), 'a')
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()
#pool.close()

