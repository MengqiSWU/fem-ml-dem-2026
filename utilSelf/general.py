import os
import pickle
import sys

import numpy as np


def check_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.mkdir(path)
            print('\t\tDirectory made as %s' % path)


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
        pool = None
    return pool


def writeLine(fname, s, mode='w'):
    f = open(fname, mode=mode)
    f.write(s)
    f.close()


def echo(*args):
    print('\n' + '=' * 80)
    for i in args:
        print('\t%s' % i)


def getCons(mode, ndim=3,  nump=1, explicitFlag=False, numg=None, **kwargs):   # pool=None, 改进后的调用多进程不用pool
    if 'vonmises' in mode:
        from FEMxEPxML.vonmisesCons import vonmisesConstitutive
        save_flag = True
        cons = vonmisesConstitutive(
            explicitFlag=explicitFlag, numg=numg, nump=nump,
            p0=kwargs['p0'], nu=kwargs['poisson'], E=kwargs['E'], rho=kwargs['rho'],
            verboseFlag=False, ndim=ndim, save_path=kwargs['save_path'], save_flag=save_flag)
    elif mode == 'eb':
        from FEMxEPxML.EBmodelCons import EBmodelConstitutive
        cons = EBmodelConstitutive(ndim=ndim, explicit_flag=explicitFlag, numg=numg, save_flag=True, **kwargs)
    elif mode == 'csuh':
        from FEMxEPxML.csuhCons import csuhConstitutive
        cons = csuhConstitutive(
            explicitFlag=explicitFlag, ndim=ndim, rho=kwargs['rho'],
            p0=kwargs['p0'],
            numg=numg, nump=nump, save_path=kwargs['save_path'], save_flag=True, **kwargs['csuh_dic'])
    elif mode == 'uh':
        from FEMxEPxML.UHcons import uhConstitutive
        cons = uhConstitutive(
            explicitFlag=explicitFlag, ndim=ndim, rho=kwargs['rho'],
            p0=kwargs['p0'], ocr=kwargs['ocr'], numg=numg, nump=nump, save_path=kwargs['save_path'])
    elif mode == 'norsand':
        from FEMxEPxML.norsandCons import NorSandConstitutive
        cons = NorSandConstitutive(explicitFlag=explicitFlag, ndim=ndim, rho=kwargs['rho'],
                                   p0=kwargs['p0'], numg=numg, nump=nump, e0=kwargs['e0'],
                                   save_path=kwargs['save_path'])
    elif mode == 'mldem':
        import torch
        from FEMxEPxML.mldemCons import MlDemConstitutive
        NN_sig = torch.load(
            kwargs['NN_sig_path'],
            map_location=torch.device('cpu'))
        NN_D = None
        if not explicitFlag:
            NN_D = torch.load(
                kwargs['NN_D_path'],
                map_location=torch.device('cpu'))
        cons = MlDemConstitutive(p0=kwargs['p0'], NN_sig=NN_sig, NN_D=NN_D, explicitFlag=explicitFlag, numg=numg, nump=nump,
                                 rho=kwargs['rho'],input_features=kwargs['input_features'], save_path=kwargs['save_path'])


    elif mode == 'mldem3d':
        import torch
        from FEMxEPxML.mldemCons3d_accum import MlDemConstitutive
        NN_sig = torch.load(
            kwargs['NN_sig_path'],
            map_location=torch.device('cpu'))

        NN_D = None
        if not explicitFlag:
            NN_D = torch.load(
                kwargs['NN_D_path'],
                map_location=torch.device('cpu'))

        cons = MlDemConstitutive(p0=kwargs['p0'], NN_sig=NN_sig, NN_D=NN_D, explicitFlag=explicitFlag, numg=numg, nump=nump,
                                 rho=kwargs['rho'], input_features=kwargs['input_features'], save_path=kwargs['save_path'])


    #
    # elif mode == 'mldem3d':
    #     import torch
    #     from FEMxEPxML.mldemCons3d_split_D import MlDemConstitutive
    #     NN_sig = torch.load(
    #         kwargs['NN_sig_path'],
    #         map_location=torch.device('cpu'))
    #
    #     NN_Dv = None
    #     if not explicitFlag:
    #         NN_Dv = torch.load(
    #             kwargs['NN_Dv_path'],
    #             map_location=torch.device('cpu'))
    #     NN_Dr = None
    #     if not explicitFlag:
    #         NN_Dr = torch.load(
    #             kwargs['NN_Dr_path'],
    #             map_location=torch.device('cpu'))
    #
    #     cons = MlDemConstitutive(p0=kwargs['p0'], NN_sig=NN_sig,  NN_Dv=NN_Dv, NN_Dr=NN_Dr, explicitFlag=explicitFlag, numg=numg, nump=nump,
    #                              rho=kwargs['rho'], input_features=kwargs['input_features'], save_path=kwargs['save_path'])


    # elif mode == 'mldem3d':
    #     import torch
    #     from FEMxEPxML.mldemCons3d_split_sig import MlDemConstitutive
    #     NN_sig = None
    #     NN_sigv = torch.load(
    #         kwargs['NN_sigv_path'],
    #         map_location=torch.device('cpu'))
    #     NN_sigr = torch.load(
    #         kwargs['NN_sigr_path'],
    #         map_location=torch.device('cpu'))
    #
    #
    #     NN_D = None
    #     if not explicitFlag:
    #         NN_D = torch.load(
    #             kwargs['NN_D_path'],
    #             map_location=torch.device('cpu'))
    #
    #     cons = MlDemConstitutive(p0=kwargs['p0'], NN_sig=NN_sig, NN_sigv=NN_sigv, NN_sigr=NN_sigr,  NN_D=NN_D, explicitFlag=explicitFlag, numg=numg, nump=nump,
    #                              rho=kwargs['rho'], input_features=kwargs['input_features'], save_path=kwargs['save_path'])




    elif mode == 'dem':
        from FEMxDEM.demCons2d import demConstitutive
        cons = demConstitutive(
            p0=kwargs['p0'], ndim=2, explicitFlag=explicitFlag, numg=numg, nump=nump, save_path=kwargs['save_path'],
            save_flag=True if explicitFlag else False,
        )

    elif mode == 'dem3d':
        from FEMxDEM.demCons3d_accum import demConstitutive
        cons = demConstitutive(
            p0=kwargs['p0'], ndim=3, explicitFlag=explicitFlag, numg=numg, nump=nump, save_path=kwargs['save_path'],
            save_flag=True if explicitFlag else False,
        )

    elif mode == 'elastic':
        from FEMxEPxML.elasticCons import elasticConstitutive
        cons = elasticConstitutive(
            E=kwargs['E'], poisson=kwargs['poisson'], rho=kwargs['rho'],
            p0=kwargs['p0'], ndim=2, explicitFlag=explicitFlag, numg=numg, save_path=kwargs['save_path'])
    elif mode == 'mixed' or mode == '2ml':
        import torch
        from FEMxEPxML.ml_with_x_Cons import MixedConstitutive
        save_path = kwargs['save_path']
        if save_path is None:
            raise
        NN_sig = torch.load(
            kwargs['NN_sig_path'],
            map_location=torch.device('cpu'))
        if 'csuh' in kwargs['NN_sig_path']:
            x_name = 'csuh'
        elif mode == '2ml':
            x_name = 'mldem'
        else:
            x_name = 'vonmises'
        cons = MixedConstitutive(
            ndim=ndim,
            save_path=save_path, NN_sig=NN_sig, input_features=kwargs['input_features'],
            rho=kwargs['rho'],
            p0=kwargs['p0'], explicitFlag=explicitFlag, numg=numg, nump=nump,
            kwargs=kwargs, x_name=x_name)
    else:
        raise ValueError('Mode %s not involved yet.' % mode)
    return cons


def pickle_dump(**kwargs):
    root_path = kwargs['root_path']
    savePath = os.path.join(root_path, 'scalar')
    check_mkdir(savePath)
    for k in kwargs:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'wb')
            pickle.dump(kwargs[k], f, 0)
            f.close()
    print('\tScalar saved in %s' % savePath)


def pickle_load(*args, root_path):
    cwd = os.getcwd()
    # if 'FEMxML' not in cwd:
    #     root_path = os.path.join(cwd, 'FEMxML')
    if 'sciptes4figures' in cwd:
        root_path = os.getcwd()
    savePath = os.path.join(root_path, 'scalar')
    # if not os.path.exists(savePath):
    #     os.mkdir(savePath)
    if 'epoch' in root_path:
        root_path = os.path.split(root_path)[0]
        savePath = os.path.join(root_path, 'scalar')
    print()
    print('-' * 80)
    print('Note: Scalar restored from %s' % savePath)
    for k in args:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'rb')
            # eval('%s = pickle.load(f)' % k)
            yield eval('pickle.load(f)')
            f.close()


def mapMask(param):
    return param[0](param[1])


def get_load_information(
        out_directory, test_name, mode, explicit_flag, order, numg,
        nx=None, ny=None, smooth_flag=None, mesh_name=None, rate_vel=None, safety_coefficient=None, vel=None, damp=None,
        **kwargs):
    temp = test_name
    if smooth_flag is not None:
        temp += '_%s' % ('smooth' if smooth_flag else 'rough')
    temp += '_%s_%s_intorder%d_numg%d' % (
        'explicit' if explicit_flag else 'implicit', mode, order, numg)
    if mesh_name:
        temp += '_%s' % mesh_name
    else:
        temp += '_x%d_y%d' % (nx, ny)
    if explicit_flag:
        if rate_vel is not None:
            temp += '_rate%.2f' % np.abs(rate_vel)
        else:
            temp += '_vel%.2f' % np.abs(vel)
        if damp is not None and damp != 0.:
            temp += '_damp%.1e' % (damp)
        temp += '_safe%.1f' % (safety_coefficient)
    if mode == 'uh':
        temp += '_p%dkPa_ocr%.1f' % (kwargs['p0'] / 1e3, kwargs['ocr'])
    elif mode == 'csuh':
        temp += '_p%dkPa_ocr_%.1f' % (kwargs['p0'] / 1e3, kwargs['ocr'])
        temp += '_theta%d' % kwargs['csuh_dic']['theta_degree']
    elif mode == 'norsand':
        temp += '_p%dkPa_e%.3f' % (kwargs['p0'] / 1e3, kwargs['e0'])
    elif mode == 'mldem' or mode == 'mixed':
        if 'active' in kwargs['NN_sig_path']:
            temp += '_active'
        temp += '_NN%s' % kwargs['nn_name']
    elif mode == 'eb':
        temp += '_fric_%.1f' % kwargs['fric']
    elif mode == '2ml':
        temp += '_%s' % kwargs['input_features']

    if explicit_flag:
        temp += "_timestep%.1e" % kwargs['time_step']



    name = os.path.join(out_directory, temp)
    return name


def get_time_step(rho, lam_2G, element_size, safety_coefficient=0.2):
    time_step = safety_coefficient * np.sqrt(rho / lam_2G) * element_size
    return time_step


def explicit_material_constants(p0=None, nn_name=None, nn_name_D=None, nn_name_sigv=None, nn_name_sigr=None, nn_name_Dv=None, nn_name_Dr=None, csuh_para_line=None, active_iter=None):
    if p0 is None:
        p0 = 1e5  # confining pressure
    else:
        p0 = p0
    # footing-dem Loss :5.459e-02 	 kappa:5.212e-02 	 lambdaa:1.488e-01 	 N:1.791e+00 	 Z:9.759e-01 	 ocr:3.599e+01 	 theta_degree:2.359e+01
    if csuh_para_line is None:
        # csuh_dic = get_dic_from_string(s='ocr:120. \t theta_degree:30. \t lambdaa:0.135 \t kappa:0.04 \t N:1.973 \t Z:0.933938655')
        # csuh_dic = get_dic_from_string(s='kappa:5.748e-02 	 lambdaa:1.500e-01 	 N:1.804e+00 	 Z:9.415e-01 	 ocr:3.207e+01 	 theta_degree:2.578e+01')  # fine
        # csuh_dic = get_dic_from_string(s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 ocr:3.774e+02 	 theta_degree:1.329e+01')  # optimized from the dataset collected from the dem simulation
        csuh_dic = get_dic_from_string(s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 ocr:3.774e+02 	 theta_degree:8.')  # optimized from the dataset collected from the dem simulation
        # csuh_dic = get_dic_from_string(s='ocr:20. \t theta_degree:30. \t lambdaa:0.1689 \t kappa:0.1 \t N:2.021 \t Z:0.9358')
    else:
        # csuh_dic = get_dic_from_string('kappa:5.111e-02 	 lambdaa:1.485e-01 	 N:1.790e+00 	 Z:9.824e-01 	 ocr:3.833e+01 	 theta_degree:2.314e+01')
        csuh_dic = get_dic_from_string(csuh_para_line)
    ocr = csuh_dic['ocr']

    poisson = 0.2

    # original

    e0 = 0.6
    E = 2e7
    lam = E * poisson / (1 + poisson) / (1 - 2 * poisson)
    G = E / 2 / (1 + poisson)
    rho = 2650  # kg/m^3

    # nn_name = 'X_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_mix_biaxial_1'
    kwargs = {'p0': p0, 'ocr': ocr, 'e0': e0, 'lam': lam, 'rho': rho, 'G': G,
              'poisson': poisson,
              'E': E,
              'csuh_dic': csuh_dic,
              }
    if nn_name is not None:
        input_features = nn_name.split('X_')[1].split('_')[0]
        if active_iter is None:
            # kwargs['NN_sig_path'] = './FEMxML/biax_ml_1e5/%s/entire_model.pt' % nn_name   #for biaxial
            # kwargs['NN_sig_path'] = './FEMxML/footing_ml/%s/entire_model.pt' % nn_name       #for footing
            kwargs['NN_sig_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name  # for 3d biaxial
        else:
            kwargs['NN_sig_path'] = './FEMxML/%s/entire_model_iter%d.pt' % (nn_name, active_iter)
        kwargs['input_features'] = input_features
        if 'active' in nn_name:
            temp = nn_name.split('/')[2]
            if active_iter is not None:
                temp += '_iter%d' % active_iter
            kwargs['nn_name'] = temp
        else:
            kwargs['nn_name'] = os.path.split(nn_name)[-1]

    if nn_name_sigv and nn_name_sigr is not None:
        input_features = nn_name_sigv.split('X_')[1].split('_')[0]
        kwargs['input_features'] = input_features
        kwargs['NN_sigv_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name_sigv
        kwargs['NN_sigr_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name_sigr


    if nn_name_D is not None:
        if active_iter is None:
            # kwargs['NN_D_path'] = './FEMxML/biax_ml_1e5/%s/entire_model.pt' % nn_name_D    #for biaxial
            # kwargs['NN_D_path'] = './FEMxML/footing_ml/%s/entire_model.pt' % nn_name_D    #for footing
            kwargs['NN_D_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name_D    #for footing
        else:
            kwargs['NN_D_path'] = './FEMxML/%s/entire_model_iter%d.pt' % (nn_name_D, active_iter)
    elif nn_name_Dv and nn_name_Dr is not None:
        kwargs['NN_Dv_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name_Dv
        kwargs['NN_Dr_path'] = './FEMxML/triax_ml_1e5/%s/entire_model.pt' % nn_name_Dr

    return p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs


def get_dic_from_string(s: str):
    dic = {}
    s = s.replace(' ', '')
    line_list = s.split('\t')
    for i in line_list:
        temp = i.split(':')
        if temp[0] in "kappa,lambdaa,N,Z,ocr,theta_degree,M":
            dic[temp[0]] = float(temp[1])
    return dic


