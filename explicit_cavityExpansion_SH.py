import os
import sys
import time
import numpy as np
from esys.escript import kronecker, FunctionOnBoundary, Function, \
    whereZero, sup, inf, interpolate, integrate, Scalar, wherePositive,sqrt
from esys.finley import ReadGmsh
from FEMxDEM.explicitSolver import explicitSolver
from utilSelf.esysUtils import force_length_calculation, boudary_explicit_2D_biaxial, plot_model, save_explicit_vtk
from utilSelf.general import echo, check_mkdir, getCons, writeLine, \
    get_load_information, get_time_step, explicit_material_constants

argvs = sys.argv
gama = None
n_iter, i = None, 0
while i < len(argvs):
    if argvs[i] == '-n':
        n_iter = int(argvs[i + 1])
    elif argvs[i] == '-gama':
        gama = float(argvs[i + 1])
    i += 1

if n_iter:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d' % n_iter
else:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_0'

# get the MPI-pool on the server
test_name = 'cavity_quater'
mpiFlag = False
nump = 4
explicitFlag = True
smoothFlag = False

# damp = 1e6
damp = 2. * gama * np.sqrt(2650 * 2e7) if gama else 1e6
mode = 'csuh'  # 'mldem' 'dem' 'elastic' 'csuh' 'uh'
# 'vonmises' 'mixed' '2ml'
# 'misesideal' 'rnn_misesideal'  rnn_misesideal_fc
# 'drucker' 'rnn_drucker' rnn_drucker_fc
#  mises_harden  rnn_mises_harden  rnn_mises_harden_fc
# rnn_mises_harden_epsp rnn_mises_harden_epsp_fc
# rnn_mises_harden_extract
# if set the scalar as a number means using this number in scalar,
# else if set scalar=0, then we using the adaptive length of the scalar
scalar = 0

if mode == 'eb':
    fric = 30.
axialStrainGoal = 0.10
time_step = 2e-4
# vel = -0.05
vel = 0.1
mesh_number = 2
nx, ny = mesh_number, mesh_number * 2

save_flag = True if mode == 'dem' or mode == 'mldem' else False
order = 1
integration_order = 1
nn_name_temp = nn_name
input_features = nn_name.split('_')[1]
output_features = nn_name.split('_')[3]

mesh_name = 'Cavity_quarter'
mydomain = ReadGmsh('./meshes/cavity_msh/%s.msh' % mesh_name, numDim=2,
                    order=order, integrationOrder=integration_order)



dim = mydomain.getDim()
k = kronecker(mydomain)
numg = len(Function(mydomain).getX().toListOfTuples())
safety_coefficient = 0.5

# cal the stable timeStep
"""
    critical time step calculation :
        dt_critical = le_min*sqrt(rho/E)

    NOTE: Timestep bigger than 0.1*dt will result in non-convergence.
"""
# load information
# material properties'
# if mode == 'csuh':

# csuh_para_line = "kappa:1.906e-01\tlambdaa:2.142e-01\tN:1.931e+00\tZ:2.743e-01\tocr:3.774e+02\ttheta_degree:1.329e+01"
# parameters inversed from the dem footing simulation

# csuh_para_line = "kappa:3.8330e-02\tlambdaa:6.9290e-02\tN:2.9070e+00\tocr:2.0870e+02\tM:3.9230e-01\tm:1.2430e+01\tnu:2.3910e-01"   # paramters inversed from the exbiaxial dem


# before
# csuh_para_line = "kappa:1.545e-01\tlambdaa:2.342e-01\tN:2.050e+00\tZ:2.522e-01\tocr:2.204e+03\tM:3.686e-01\tm:4.053e-01\tnu:1.997e-01"
# csuh_para_line = "kappa:0.04\tlambdaa:0.135\tN:1.9\tZ:1.9\tocr:1.2+e02\tM:1.25\tm:1.8\tnu:2.0e-01"


# after
# csuh_para_line = "kappa:3.3450e-01\tlambdaa:3.4360e-01\tN:2.6870e+00\tZ:-7.7820e-01\tocr:9.1100e+02\tM:6.2040e-01\tm:3.2080e+01\tnu:2.4340e-01"   # paramters inversed from the exfooting dem

# csuh_para_line = 'kappa:3.2860e-01\tlambdaa:3.4220e-01\tN:2.7080e+00\tZ:-7.6940e-01\tocr:7.8780e+02\tM:6.2960e-01\tm:2.9260e+01\tnu:2.6320e-01'
# csuh_para_line = 'kappa:2.592e-01\tlambdaa:2.830e-01\tN:2.334e+00 \tZ: -6.198e-03\tocr:9.624e+02\tM:4.442e-01\tm:7.315e-01\tnu:2.296e-01'
# csuh_para_line = 'kappa:2.418e-01\tlambdaa:2.588e-01\tN:2.238e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:4.010e-01\tm:4.424e-01\tnu:2.333e-01'

if mode == 'csuh':

    # csuh_para_line = "kappa:1.906e-01\tlambdaa:2.142e-01\tN:1.931e+00\tZ:2.743e-01\tocr:3.774e+02\ttheta_degree:1.329e+01"
    # parameters inversed from the dem footing simulation

    # csuh_para_line = "kappa:3.8330e-02\tlambdaa:6.9290e-02\tN:2.9070e+00\tocr:2.0870e+02\tM:3.9230e-01\tm:1.2430e+01\tnu:2.3910e-01"   # paramters inversed from the exbiaxial dem

    # before
    # csuh_para_line = "kappa:1.545e-01\tlambdaa:2.342e-01\tN:2.050e+00\tZ:2.522e-01\tocr:2.204e+03\tM:3.686e-01\tm:4.053e-01\tnu:1.997e-01"
    # csuh_para_line = "kappa:0.04\tlambdaa:0.135\tN:1.9\tZ:1.9\tocr:1.2+e02\tM:1.25\tm:1.8\tnu:2.0e-01"

    # after
    # csuh_para_line = "kappa:3.3450e-01\tlambdaa:3.4360e-01\tN:2.6870e+00\tZ:-7.7820e-01\tocr:9.1100e+02\tM:6.2040e-01\tm:3.2080e+01\tnu:2.4340e-01"   # paramters inversed from the exfooting dem

    # csuh_para_line = 'kappa:3.2860e-01\tlambdaa:3.4220e-01\tN:2.7080e+00\tZ:-7.6940e-01\tocr:7.8780e+02\tM:6.2960e-01\tm:2.9260e+01\tnu:2.6320e-01'

    # csuh_para_line = 'kappa:2.592e-01\tlambdaa:2.830e-01\tN:2.334e+00 \tZ: -6.198e-03\tocr:9.624e+02\tM:4.442e-01\tm:7.315e-01\tnu:2.296e-01'
    # csuh_para_line = 'kappa:2.418e-01\tlambdaa:2.588e-01\tN:2.238e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:4.010e-01\tm:4.424e-01\tnu:2.333e-01'
    # csuh_para_line = 'kappa:2.418e-01\tlambdaa:2.588e-01\tN:2.238e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:3.450e-01\tm:4.424e-01\tnu:2.333e-01' #变差
    # csuh_para_line = 'kappa:2.418e-01\tlambdaa:2.588e-01\tN:2.338e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:4.450e-01\tm:4.424e-01\tnu:2.333e-01' #变差
    # csuh_para_line = 'kappa:2.318e-01\tlambdaa:2.588e-01\tN:2.338e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:4.250e-01\tm:4.424e-01\tnu:2.333e-01' #变好一点，前期可以，后期变窄
    # csuh_para_line = 'kappa:2.318e-01\tlambdaa:2.588e-01\tN:2.338e+00 \tZ: 1.219e-01\tocr:9.906e+02\tM:4.150e-01\tm:4.224e-01\tnu:2.333e-01'  # 变好一点，前期可以，后期变窄

    # csuh_para_line = 'kappa:2.406e-01\tlambdaa:2.655e-01\tN:2.240e+00 \tZ: 9.647e-02\tocr:9.755e+02\tM:4.396e-01\tm:4.822e-01\tnu:2.289e-01'
    # csuh_para_line = "kappa:2.615e-01\tlambdaa:2.882e-01\tN:2.364e+00\tZ:-4.863e-02\tocr:9.237e+02\tM:3.450e-01\tm:6.932e-01\tnu:2.244e-01"  # paramters inversed from the exfooting dem good

    # csuh_para_line = "kappa:2.401e-01\tlambdaa:2.649e-01\tN:2.233e+00\tZ:1.010e-01\tocr:9.403e+02\tM:3.450e-01\tm:6.932e-01\tnu:2.293e-01"  # paramters inversed from the exfooting dem （step=48900 with revised）
    # csuh_para_line = "kappa:2.401e-01\tlambdaa:2.649e-01\tN:2.353e+00\tZ:1.010e-01\tocr:9.403e+02\tM:3.450e-01\tm:6.932e-01\tnu:2.293e-01"  # paramters inversed from the exfooting dem （step=48900 with revised1 closer）
    # csuh_para_line = "kappa:2.401e-01\tlambdaa:2.649e-01\tN:2.453e+00\tZ:1.010e-01\tocr:9.403e+02\tM:3.350e-01\tm:6.732e-01\tnu:2.293e-01"  # paramters inversed from the exfooting dem （step=48900 with revised2 closerer）
    # csuh_para_line = "kappa:2.401e-01\tlambdaa:2.649e-01\tN:2.553e+00\tZ:1.010e-01\tocr:9.403e+02\tM:4.850e-01\tm:4.032e-01\tnu:2.293e-01"  # paramters inversed from the exfooting dem （step=48900 with revised2 closerer）

    # csuh_para_line = "kappa:2.256e-01\tlambdaa:2.506e-01\tN:2.152e+00\tZ:1.733e-01 \tocr:1.253e+03\tM:4.025e-01\tm:4.052e-01\tnu:2.247e-01"  # paramters inversed from the exfooting dem （step=22500 with revised3 ）
    # csuh_para_line = "kappa:2.256e-01\tlambdaa:2.456e-01\tN:2.252e+00\tZ:1.733e-01 \tocr:1.253e+03\tM:4.025e-01\tm:4.052e-01\tnu:2.247e-01"  # paramters inversed from the exfooting dem （step=22500 with revised3 closer）
    # csuh_para_line = "kappa:2.306e-01\tlambdaa:2.456e-01\tN:2.302e+00\tZ:1.733e-01 \tocr:1.253e+03\tM:2.525e-01\tm:3.052e-01\tnu:2.247e-01"  # paramters inversed from the exfooting dem （step=22500 with revised3 closer）

    # csuh_para_line = "kappa:1.545e-01\tlambdaa:2.378e-01\tN:2.124e+00\tZ:2.541e-01 \tocr:2.183e+03\tM:3.686e-01\tm:4.053e-01\tnu:1.998e-01"  # inversed from the exfooting dem original
    # csuh_para_line = "kappa:1.545e-01\tlambdaa:2.378e-01\tN:2.124e+00\tZ:2.541e-01 \tocr:1.203e+03\tM:3.686e-01\tm:3.753e-01\tnu:2.638e-01"
    # csuh_para_line = "kappa:1.545e-01\tlambdaa:2.578e-01\tN:2.124e+00\tZ:2.741e-01 \tocr:1.403e+03\tM:3.686e-01\tm:4.553e-01\tnu:2.638e-01"

    # csuh_para_line = "kappa:2.094e-01\tlambdaa:2.361e-01\tN:2.070e+00\tZ:2.375e-01 \tocr:1.702e+03\tM:3.661e-01\tm:4.016e-01\tnu:2.225e-01"
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.421e-01\tN:2.170e+00\tZ:2.675e-01 \tocr:1.802e+03\tM:3.681e-01\tm:3.956e-01\tnu:2.225e-01"   # better
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:2.781e-01 \tocr:1.780e+03\tM:3.681e-01\tm:3.906e-01\tnu:2.225e-01"    # similar
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:2.881e-01 \tocr:1.750e+03\tM:3.681e-01\tm:3.906e-01\tnu:2.630e-01" # similar 但是剪切带稍微宽了一点点
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:2.881e-01 \tocr:1.680e+03\tM:3.610e-01\tm:3.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:2.881e-01 \tocr:1.280e+03\tM:3.610e-01\tm:3.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.070e+00\tZ:2.881e-01 \tocr:1.080e+03\tM:3.610e-01\tm:3.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:1.870e+00\tZ:2.881e-01 \tocr:9.80e+02\tM:3.610e-01\tm:3.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现，减小ocr和N行不通
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:3.881e-01 \tocr:1.679e+03\tM:3.610e-01\tm:3.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现 但是增加Z好像没什么影响
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:3.881e-01 \tocr:1.678e+03\tM:3.510e-01\tm:6.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现 但是增加m,调小M好像没什么影响
    # csuh_para_line = "kappa:1.994e-01\tlambdaa:2.425e-01\tN:2.170e+00\tZ:3.881e-01 \tocr:1.677e+03\tM:3.510e-01\tm:6.756e-01\tnu:2.630e-01" # similar 剪切带依旧延后出现 增加kappa还是不太行
    csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.453e+00\tZ:3.881e-01 \tocr:9.40e+02\tM:3.350e-01\tm:6.732e-01\tnu:2.293e-01"  # 剪切带提前出现，借用了之前部分参数 （完美）
    # csuh_para_line = "kappa:1.894e-01\tlambdaa:2.425e-01\tN:2.453e+00\tZ:4.281e-01 \tocr:9.401e+02\tM:3.350e-01\tm:6.732e-01\tnu:2.293e-01" # 剪切带提前出现，借用了之前部分参数 （依然可以）

else:
    csuh_para_line = None

confining = 1e5  # confining pressure
if (mode == 'mises_harden') or (mode == 'misesideal') or (mode == 'rnn_misesideal') or (mode == 'rnn_misesideal_fc') \
        or (mode == 'rnn_mises_harden') or \
        (mode == 'rnn_mises_harden_fc') or (mode == 'rnn_mises_harden_epsp') or (mode == 'rnn_mises_harden_epsp_fc') \
        or (mode == 'rnn_mises_harden_extract'):
    confining = 0.

p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining, nn_name=nn_name,
        csuh_para_line=csuh_para_line)
kwargs['input_features'] = input_features
kwargs['output_features'] = output_features
kwargs['b_flag'] = True

if 'rnn' in mode:
    kwargs['scalar'] = scalar

# input args
input_args = sys.argv

n, argv_len = 0, len(input_args)
while n < argv_len:
    if '-ocr' in input_args[n]:
        kwargs['ocr'] = float(input_args[n + 1])
        ocr = kwargs['ocr']
    if '-fric' in input_args[n]:
        kwargs['fric'] = float(input_args[n + 1])
        fric = kwargs['fric']
    if '-mode' in input_args[n]:
        mode = input_args[n + 1]
    n += 1

rateVelocity = vel / ly
out_directory = '../simu/explicit/%s' % test_name




loadInfor = get_load_information(
    out_directory=out_directory, test_name=test_name, mode=mode,
    explicit_flag=explicitFlag, time_step=time_step,
    vel=vel, nx=nx, ny=ny, order=order, numg=numg,
    safety_coefficient=safety_coefficient, mesh_name=mesh_name, damp=damp, gama=gama, **kwargs)

# loadInfor += '_deepfc_adaptstep'
loadInfor += 'NEW'

kwargs['save_path'] = loadInfor
timeStep = get_time_step(
    lam_2G=lam + 2 * G, rho=rho, element_size=inf(mydomain.getSize()),
    safety_coefficient=safety_coefficient)

check_mkdir(
    out_directory,
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'added_points'),
    os.path.join(loadInfor, 'iteration_gauss'), )

# plot_model(domain=mydomain, order=order, integration_order=integration_order, save_path=loadInfor)

# ---------------------- echo ------------------------
echo(
    loadInfor,
    'The stable timeStep: %e' % timeStep,
    'Solving mode %s' % ('single' if nump == 0 else ('python multiprocess %d' % nump)),
    'Explicit-%s' % (mode),
    'CWD %s' % loadInfor,
    'Solution Mode: \t%s' % ('Explicit' if explicitFlag else 'Implicit'),
    # 'lx:\t%.5f' % lx + '\t nx:\t%d' % nx,
    # 'ly:\t%.5f' % ly + '\t ny:\t%d' % ny,
    'confing:\t%e' % confining,
    # 'Axial strain:\t%f' % axialStrainGoal
)

cons = getCons(mode, numg=numg, nump=nump, explicitFlag=explicitFlag, ndim=2, **kwargs)

prob = explicitSolver(domain=mydomain, timestep=timeStep, cons=cons, loadInfor=loadInfor,
                      save_flag=save_flag, domain_size=lx * ly, damp=damp)






#----------------------------------------------------------------------------------#
# x = mydomain.getX()  # nodal coordinate
# bx = FunctionOnBoundary(mydomain).getX()
# # fx means the function on boundary, while n means the node
# fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))
# fy_, fy = whereZero(bx[1] - inf(bx[1])), whereZero(bx[1] - sup(bx[1]))
# nx_, nx = whereZero(x[0] - inf(x[0])), whereZero(x[0] - sup(x[0]))
# ny_, ny = whereZero(x[1] - inf(x[1])), whereZero(x[1] - sup(x[1]))
#
# forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fy)
# fname = os.path.join(loadInfor, 'biaxial_surf.dat')
# writeLine(fname=fname, s='AxialStrain forceTop lengthTop volumeStrain aMax iterNum ke pe work workout\n', mode='w')
# # writeLine(fname=fname, s='%.6f %.1f %.6f %.6f %.5f %d' % (0., -5e3, lengthTop, 0., 0., 0) + '\n', mode='a')
# # Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom
# if smoothFlag:
#     Dbc = ny * [0, 1] + \
#           ny_ * [0, 1] + whereZero(x[0] - .5 * lx) * [1, 1]  # bind the mind point in order not to slide in x direction
# else:
#     Dbc = ny * [1, 1] + ny_ * [1, 1]  # bind the mind point in order not to slide in x direction
#
# # Dirichlet BC values NOTE: if use the explicit format, the boundary value is for the accelerations.
# # On the top and the bottom surface, the accelerations on direction y should be 0
# Vbc = nx * [0, 0] + \
#       nx_ * [0, 0]  # bind the mind point in order not to slide in x direction
# Nbc = fx_ * [confining, 0] + \
#       fx * [-confining, 0]
#----------------------------------------------------------------------------------#



x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()

FSb = FunctionOnBoundary(mydomain)
FSx = x.getFunctionSpace()

nx  = whereZero(x[0]-inf(x[0]))              # mask左侧边界点
ny  = whereZero(x[1]-inf(x[1]))              # mask底部边界
fx  = whereZero(bx[0] - inf(bx[0]))          # mask左侧边界内的高斯点
fy  = whereZero(bx[1] - inf(bx[1]))          # mask下侧边界内的高斯点 （被mask的地方值为1）

tag_inner = mydomain.getTag("inner_boundary")  # 1
tag_outer = mydomain.getTag("outer_boundary")  # 2

n_in  = Scalar(0.0, FSx);n_in.setTaggedValue(tag_inner, 1.0)  # mask内圆边界点
n_out = Scalar(0., FSx); n_out.setTaggedValue(tag_outer, 1.0) # mask外圆边界点

f_in  = Scalar(0., FSb); f_in.setTaggedValue(tag_inner, 1.0)    #mask内圆边界内高斯点
f_out = Scalar(0., FSb); f_out.setTaggedValue(tag_outer, 1.0)   #mask外圆边界内高斯点

forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=f_in)
fname = os.path.join(loadInfor, 'Rc_surf.dat')

# 固定外圆交点：
# p_x = n_out * nx     # 外圆 ∩ x=0  => (0, rout)
# p_y = n_out * ny     # 外圆 ∩ y=0  => (rout, 0)


# 固定外圆上方交点：
lr = sup(x[0])
p_x =  n_out * wherePositive(x[1] - lr * 0.996)     # 外圆 ∩ x=0  => (0, rout)
p_y = n_out * ny*0

out_normal=prob.domain.getNormal()


q = nx * [1, 0] + ny * [0, 1] + n_in * [1, 1] + p_x * [0, 1] + p_y * [1, 0]
# q = nx * [1, 0] + ny * [0, 1] + n_in * [1, 1]
r = n_in * x * (vel / (sqrt(x[0] * x[0] + x[1] * x[1])))

# r = n_in * x * (vel/rc)
y = -confining * f_out * out_normal







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
work, work_out = 0., 0.

factor = lx * ly / np.sum((prob.domain.getSize() ** 2).toListOfTuples())
x = prob.domain.getX()
fx_double = fx + fx_
while abs(u_loaded) < ly * axialStrainGoal:  # apply 100 load steps
    # if prob.cons.cons is not None:
    #     lam_2G = np.max([prob.cons.cons[i].D[0, 0, 0, 0] for i in range(numg)])
    # else:
    #     lam_2G = lam+2.*G
    # element_size = inf(prob.domain.getSize())
    # time_step = get_time_step(
    #     lam_2G=lam_2G, rho=rho, element_size=inf(mydomain.getSize()), safety_coefficient=safety_coefficient)

    du = rateVelocity * ly * time_step
    u_loaded += du
    forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fy)
    work += du * forceTop[1]
    du_coord_ = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    du_coord = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    work_out += \
        integrate(fx_ * du_coord * [-confining, 0.] + fx * du_coord * [confining, 0.], FunctionOnBoundary(prob.domain))[
            0]
    x = prob.domain.getX()

    duBoundary = boudary_explicit_2D_biaxial(domain=prob.domain, du=du, q=ny, mapFlag=True)

    # renew the domain according to the first boundary condition
    prob.solveExplicit(n=n, duBoundary=duBoundary, dt=time_step)
    # save the the information to file loadInformation/results

    if abs(u_loaded) >= i * ly * axialStrainGoal / num_step:
        i += 1
        axialStrainCurrent = u_loaded / ly
        aMax = sup(prob.a)
        total_volume_strain = np.average(np.array(prob.volume.toListOfTuples()))

        ke = np.sum(np.array((prob.element_size * prob.energy_kinetic).toListOfTuples()))
        pe = np.sum(np.array((prob.element_size * prob.energy_potential).toListOfTuples()))

        # save the macro information
        writeLine(fname=fname,
                  s='%.6f %.1f %.6f %.6f %.5f %d %.3f %.3f %.3f %.3f' % \
                    (axialStrainCurrent, forceTop[1], lengthTop, (total_volume_strain), aMax, 0, ke, pe, work,
                     work_out) + '\n',
                  mode='a')
        echo(
            'Loading step # %d \taxialCurrentStrain: %e/%e  time_increment %e, Topforce: %.3e maxA: %.3e Time consuming: %.2e mins' %
            (n, axialStrainCurrent, axialStrainGoal, time_step, forceTop[1], aMax,
             (time.time() - time_start) / 60.))
        save_explicit_vtk(prob=prob, step=n, save_path=loadInfor, test_name=test_name, smooth_flag=smoothFlag,
                          mode=mode)
    n += 1

time_elapse = time.time() - time_start
writeLine(fname=fname, mode='a', s="# Elapsed time in hours: %.2e\n" % (time_elapse / 3600.))
