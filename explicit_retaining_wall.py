import os
import time
import numpy as np
from esys.escript import kronecker, FunctionOnBoundary, Function, \
    whereZero, sup, trace, symmetric, inner, \
    sqrt, inf, Vector, Solution, grad, whereNegative, whereNonPositive, wherePositive
from esys.finley import ReadGmsh, Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
from FEMxDEM.explicitSolver import explicitSolver
from utilSelf.esysUtils import force_length_calculation, boundary_explicit_2D_retaining, plot_model, save_explicit_vtk
from utilSelf.general import echo, check_mkdir, getCons, get_pool, writeLine, get_load_information, get_time_step,explicit_material_constants
import sys


n_iter, i = None, 0
while i < len(sys.argv):
    if sys.argv[i] == '-n':
        n_iter = int(sys.argv[i+1])
        break
    i += 1

if n_iter:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d' % n_iter
else:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_0'
# get the MPI-pool on the server
test_name = 'retaining'
mpiFlag = False
threads = 4
pool = get_pool(mpi=mpiFlag, threads=threads)
explicitFlag = True
damp = 1e6
mode = 'mldem'  # 'mldem' 'dem' 'elastic' 'csuh' 'uh' 'vonmises' 'vonmises_save'
axialStrainGoal = 0.06
# vel = -0.1
vel = -0.1
time_step = 4e-4
save_flag = True if mode == 'dem' or mode == 'mldem' else False

mesh_number = 2
lx, ly = 4., 2.
nx, ny = mesh_number * 2, mesh_number
base_height = 0.4

order = 1
integration_order = 1
nn_name_temp = nn_name
input_features = nn_name.split('_')[1]
output_features = nn_name.split('_')[3]
# input_features = 'epsANDqH'
# output_features = 'sigANDH'
mesh_name = 'retaining_321'
mydomain = ReadGmsh('./meshes/retaining_msh/%s.msh' % mesh_name, numDim=2,
                    order=order, integrationOrder=integration_order)

dim = mydomain.getDim()
k = kronecker(mydomain)
numg = len(Function(mydomain).getX().toListOfTuples())
nump = threads  # number of processes for multiprocessing
safety_coefficient = 0.5

# cal the stable timeStep
"""
    critical time step calculation :
        dt_critical = le_min*sqrt(rho/E)

    NOTE: Timestep bigger than 0.1*dt will result in non-convergence.
"""
# load information
# material properties'
# csuh_para_line= 'kappa:9.404e-02 	 lambdaa:1.611e-01 	 N:1.840e+00 	 Z:7.772e-01 	 ocr:4.810e+01 	 theta_degree:2.249e+01'
confining = 1e5  # confining pressure
p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining, nn_name=nn_name,
        csuh_para_line=None)
kwargs['input_features'] = input_features
kwargs['output_features'] = output_features
# time step definetion

rateVelocity = vel/lx
out_directory = '../simu/explicit/%s' % test_name
loadInfor = get_load_information(
    out_directory=out_directory, test_name=test_name, mode=mode, explicit_flag=explicitFlag,
    vel=vel, nx=nx, ny=ny, order=order, numg=numg,
    safety_coefficient=safety_coefficient, mesh_name=mesh_name, damp=damp, time_step=time_step, **kwargs)
kwargs['save_path'] = loadInfor
# time step definetion
timeStep = get_time_step(
    lam_2G=lam + 2 * G, rho=rho, element_size=inf(mydomain.getSize()),
    safety_coefficient=safety_coefficient)

check_mkdir('../simu/explicit')
out_directory = '../simu/explicit/%s' % test_name

check_mkdir(
    out_directory,
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'added_points'),
    os.path.join(loadInfor, 'iteration_gauss'), )

plot_model(domain=mydomain, order=order, integration_order=integration_order, save_path=loadInfor)

# ---------------------- echo ------------------------
echo(
    loadInfor,
    'The stable timeStep: %e' % timeStep,
    'Solving mode %s' % ('single' if threads == 0 else ('python multiprocess %d' % threads)),
    'Explicit-%s' % (mode),
    'CWD %s' % loadInfor,
    'Solution Mode: \t%s' % ('Explicit' if explicitFlag else 'Implicit'),
    'confing:\t%e' % confining,
    'Axial strain:\t%f' % axialStrainGoal)

# give constitutive model to the multiscale class

cons = getCons(mode, numg=numg, pool=pool, explicitFlag=explicitFlag, ndim=2, **kwargs)
# prob
prob = explicitSolver(domain=mydomain, timestep=timeStep, cons=cons, loadInfor=loadInfor, pool=pool,
                      save_flag=save_flag, domain_size=lx*ly, damp=damp)

x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()

# fx means the function on boundary, while n means the node
fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))
fy_, fy = whereZero(bx[1] - inf(bx[1])), whereZero(bx[1] - sup(bx[1]))
nx_, nx = whereZero(x[0] - inf(x[0])), whereZero(x[0] - sup(x[0]))
ny_, ny = whereZero(x[1] - inf(x[1])), whereZero(x[1] - sup(x[1]))
base_n = whereNonPositive(x[1]-base_height)*nx
load_n = wherePositive(x[1]-base_height)*nx
load_f = wherePositive(bx[1]-base_height)*fx

force, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=load_f)
fname = os.path.join(loadInfor, 'biaxial_surf.dat')
writeLine(fname=fname, s='AxialStrain force lengthTop volumeStrain aMax iterNum time\n', mode='w')
echo(
    'Loading step # %d \taxialCurrentStrain: %e/%e  time_increment %e, wall_force: %.3e maxA: %.3e Time consuming: %.2e mins' %
    (0, 0., axialStrainGoal, timeStep, force[0], 0., 0.))
# writeLine(fname=fname, s='%.6f %.1f %.6f %.6f %.5f %d' % (0., -5e3, lengthTop, 0., 0., 0) + '\n', mode='a')

# Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom
q = nx_ * [1, 0] + ny_*[1, 1]+ load_n * [1, 0]+base_n*[1, 1]
r = nx_ * [0, 0] + ny_ * [0, 0] + load_n*[0, 0] + base_n*[0, 0]
y = fy * [0, -confining]
prob.initialize(y=y, q=q, r=r, D=kronecker(mydomain) * rho)

"""
    Rate loading is applied on the axial direction.

    The loading rate -100, means du_per_step = -100*timestep*ly = -0.000222907. 
"""
loadStepMax = abs(int(axialStrainGoal / rateVelocity / timeStep))
u_loaded = 0.
i, num_step = 0, 200
time_start = time.time()
n, t = 0, 0
while abs(u_loaded) < lx * axialStrainGoal:  # apply 100 load steps
    # if prob.cons.cons is not None:
    #     lam_2G = np.max([prob.cons.cons[i].D[0, 0, 0, 0] for i in range(numg)])
    # else:
    #     lam_2G = lam+2.*G
    # element_size = inf(prob.domain.getSize())
    # time_step = get_time_step(
    #     lam_2G=lam_2G, rho=rho, element_size=inf(mydomain.getSize()), safety_coefficient=safety_coefficient)
    du = rateVelocity * lx * time_step
    u_loaded += du
    duBoundary = boundary_explicit_2D_retaining(
        domain=prob.domain, du=du, q=load_n, base_height=base_height, mapFlag=True)

    # renew the domain according to the first boundary condition
    prob.solveExplicit(n=n, duBoundary=duBoundary, dt=time_step)
    # save the the information to file loadInformation/results
    if abs(u_loaded) >= i * lx * axialStrainGoal / num_step:
        i += 1
        axialStrainCurrent = u_loaded / lx
        aMax = sup(prob.a)
        force, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=load_f)
        total_volume_strain = np.average(np.array(prob.volume.toListOfTuples()))
        echo(
            'Loading step # %d \taxialCurrentStrain: %e/%e time_increment %e, wall_force: %.3e maxA: %.3e Time consuming: %.2e mins' %
            (n, axialStrainCurrent, axialStrainGoal, time_step, force[0], aMax,
             (time.time() - time_start) / 60.))

        # save the macro information
        writeLine(fname=fname,
                  s='%.6f %.1f %.6f %.6f %.5f %d %.2e' %
                    (axialStrainCurrent, force[0], lengthTop, total_volume_strain, aMax, 0,
                     (time.time()-time_start)/60) + '\n',
                  mode='a')
        save_explicit_vtk(prob=prob, save_path=loadInfor, test_name=test_name, smooth_flag=False, step=n, mode=mode)
    n += 1

time_elapse = time.time() - time_start
writeLine(fname=fname, mode='a', s="# Elapsed time in mins: %.2e\n" % (time_elapse / 60.))

if pool is not None:
    pool.close()
