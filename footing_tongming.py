import os
import time
import sys
from esys.escript import whereZero, FunctionOnBoundary, kronecker, sup, \
    symmetric, trace, sqrt, inner, whereNonPositive, wherePositive, inf, Function, Tensor
from esys.escript.pdetools import Projector
from esys.finley import ReadGmsh
from esys.weipa import saveVTK

from FEMxDEM.implicitSolver import ImplicitSolver
from utilSelf.esysUtils import force_length_calculation, boudaryConditions_footing, plot_model
from utilSelf.general import check_mkdir, writeLine, echo, getCons, get_pool, explicit_material_constants, \
    get_load_information


# ======================================================
''' 
    -numg_net 3618 -num_mesh 1206 -ratio 1.0 -integration_order 1 -active_flag 1 -iter_max 40
'''

arg_list = sys.argv
i = 0
numg_net_arg = 254
num_mesh = 303
nump = 4
ratio = None
integration_order = 1
active_flag = False
active_iter = None
iter_max = 40
while i < len(arg_list):
    if '-numg_net' in arg_list[i]:
        numg_net_arg = int(arg_list[i+1])
        echo('Input -numg_net %d' % numg_net_arg)
    if '-num_mesh' in arg_list[i]:
        num_mesh = int(arg_list[i+1])
        echo('Input -num_mesh %d' % num_mesh)
    if '-ratio' in arg_list[i]:
        ratio = float(arg_list[i+1])
        echo('Input -ratio %.2f' % ratio)
    if '-integration_order' in arg_list[i]:
        integration_order = int(arg_list[i+1])
        echo('Input -integration_order %d' % integration_order)
    if '-active_flag' in arg_list[i]:
        active_flag = True if int(arg_list[i+1]) else False
        echo('Input -active_flag %d' % active_flag)
    if '-active_iter' in arg_list[i]:
        active_iter = int(arg_list[i+1])
        echo('Input -active_iter %d' % active_iter)
    if '-iter_max' in arg_list[i]:
        iter_max = int(arg_list[i+1])
        echo('Input -iter_max %d' % iter_max)
    i += 1

# ======================================================
time_start = time.time()
test_name = 'footing'
out_directory = '../simu/%s' % test_name
mode = 'csuh'  # 'csuh' 'mldem' 'mises' 'dem' 'norsand'
surcharge = p0 = confining = 1e5  # surcharge equals to the initial vertical stress of the RVE packing
ndim = 2
B, L, H = 0.25, 4., 2.
strainObject = 0.12
loadStep = 400
vel = strainObject * H / loadStep

# read Gmsh mesh with 6-node triangle element (2500 tri6); each element has 3 Gauss points
order = 1
mshname = 'footing%d' % num_mesh
mydomain = ReadGmsh(
    './/meshes/footing_msh/%s.msh' % mshname, numDim=ndim,
    order=order, integrationOrder=integration_order)
kronecker_ = kronecker(mydomain)
numg = len(Function(mydomain).getX().toListOfTuples())  # number of Gauss points

# ==============================================
# material properties
if mode == 'mldem':
    if active_flag:
        NN_sig_path = 'footing_ml/active_footing_%d/X_epsANDabsxy_Y_sig_numNN3_dd5/active_0' % numg_net_arg
        NN_D_path = 'footing_ml/active_footing_%d/X_epsANDabsxy_Y_D_numNN3_dd8/active_0' % numg_net_arg
        # NN_sig_path = 'footing_ml/active_footing_%d_6_after/X_epsANDabsxy_Y_sig_dd5_noFourier_noRotate_after_resample' % numg_net_arg
        # NN_D_path = 'footing_ml/active_footing_%d_6_after/X_epsANDabsxy_Y_D_dd8_noFourier_noRotate_after_resample' % numg_net_arg
        numg_net = numg_net_arg
    else:
        NN_sig_path = 'footing_ml/X_epsANDabsxy_Y_sig_dd5_noFourier_noRotate_%d' % numg_net_arg
        NN_D_path = 'footing_ml/X_epsANDabsxy_Y_D_dd8_noFourier_noRotate_%d' % numg_net_arg
        numg_net = int(NN_sig_path.split('_')[-1])
    if ratio is not None and active_flag is False:
        NN_sig_path += '_ratio%.1f' % ratio
        NN_D_path += '_ratio%.1f' % ratio
else:
    NN_sig_path = None
    NN_D_path = None

csuh_line = None
# csuh_line = 'kappa:2.333e-01 	 lambdaa:2.551e-01 	 N:2.182e+00 	 Z:1.009e-01 	 ocr:4.685e+02 	 theta_degree:1.278e+01'
p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining,
        csuh_para_line=csuh_line,
        nn_name=NN_sig_path,
        nn_name_D=NN_D_path,
        active_iter = active_iter,
    )

loadInfor = get_load_information(out_directory=out_directory, test_name=test_name, mode=mode,
                                 explicit_flag=False, order=integration_order, numg=numg, mesh_name=mshname, **kwargs)
# loadInfor += '_first_generation'
kwargs['save_path'] = loadInfor

echo(
    'CWD:                %s' % loadInfor,
    'Mode:               %s' % mode,
    'Num Porcess:        %d' % nump,
    'Mesh infor:         %s' % mshname,
    'Order_integration:  %d' % integration_order,
    'Num Guass:          %d' % numg,
    'Confining:          %.3e' % p0,
)

pool = get_pool(mpi=False, threads=nump)

check_mkdir(
    loadInfor,
    os.path.join(loadInfor, 'gauss'),
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'iteration_gauss'),
    os.path.join(loadInfor, 'added_points'),
)

plot_model(domain=mydomain, order=order, integration_order=integration_order, save_path=loadInfor)

cons = getCons(mode, numg=numg, pool=pool, explicitFlag=False, **kwargs)

prob = ImplicitSolver(domain=mydomain, cons=cons,
                      pool=pool, loadInfor=loadInfor,
                      save_loading_flag=True if mode == 'dem' else False)
# save the initial datasets
prob.save_loading_mask(t=0, sig_data=prob.sig, D_data=prob.D, u_grad=Tensor(0., Function(prob.domain)))

x = mydomain.getX()
bx = FunctionOnBoundary(mydomain).getX()
footingBase = whereZero(bx[1] - sup(bx[1])) * whereNonPositive(bx[0] - B)
forceFoot, lengthFoot = force_length_calculation(
    sig=prob.sig, domain=mydomain, where=footingBase)
s = 'dH/H forceTopX forceTopY lengthFoot time(mins)\n' + \
    '%.3e %.3e %.3e %.3e %.3e\n' % \
    ((0.) * (-vel) / H, forceFoot[0], forceFoot[1], lengthFoot,
     (time.time() - time_start) / 60.)
echo(s)
fname = os.path.join(loadInfor, 'bearing.dat')
fout = writeLine(fname=fname, s=s, mode='w')

nx_, nx = whereZero(x[0]), whereZero(x[0] - sup(x[0]))
ny_, ny = whereZero(x[1] - inf(x[1])), whereZero(x[1] - sup(x[1]))
where_load = ny * whereNonPositive(x[0] - B)
by = whereZero(bx[1] - sup(bx[1]))
where_surcharge = by * wherePositive(bx[0] - B)

t = 0
while t < loadStep:  # apply 58 loading step; further loading would abort the program due to severe mesh distortion
    print('\n' + '-' * 20 +
          'TIME %d/%d time_consumed %.1emins ' %
          (t, loadStep, (time.time() - time_start) / 60.) + '-' * 20)
    q, r, y, Y = boudaryConditions_footing(
        nx_=nx_, ny_=ny_, nx=nx, ny=ny, by=by,
        where_load=where_load, where_surcharge=where_surcharge,
        domain=prob.domain, confining=surcharge, vel=vel, normal_flag=False)
    prob.initialize(y=y, q=q, r=r)
    prob.solve(iter_max=iter_max, t=t)

    disp = prob.u
    stress = prob.sig
    forceFoot, lengthFoot = force_length_calculation(
        sig=prob.sig, domain=prob.domain, where=footingBase)
    s = '%.3e %.3e %.3e %.3e %.3e\n' % \
        ((t + 1) * (-vel) / H, forceFoot[0], forceFoot[1], lengthFoot,
         (time.time() - time_start) / 60.)
    writeLine(fname=fname, s=s, mode='a')
    echo('Epsilon \tXforce \tYforce \tlengthFoot \ttime:' ,
         s)

    strain = prob.eps
    tangent = prob.D
    proj = Projector(mydomain, reduce=False)
    stress = proj(stress)
    strain = proj(strain)

    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * kronecker_ / ndim
    shear = sqrt(2 * inner(dev_strain, dev_strain))
    volume_strain = trace(strain)
    if mode == 'mldem':
        vtkFileName = os.path.join(loadInfor, 'vtk/footing_%s_numg%d_net%d_%d.vtu' % (mode, numg, numg_net, t))
    else:
        vtkFileName = os.path.join(loadInfor, 'vtk/footing_%s_numg%d_%d.vtu' % (mode, numg, t))
    saveVTK(vtkFileName, disp=disp, stress=stress, shear=shear, vol_eps=volume_strain)
    t += 1

# prob.getCurrentPacking(pos=packNo, time=t, prefix='./result/packing/')  # output packing
time_elapse = time.time() - time_start
writeLine(fname=fname, s="#Elapsed time in hours: %s\n" % str(time_elapse / 3600.), mode='a')

pool.close()
