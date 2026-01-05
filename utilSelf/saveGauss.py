import os

def saveGauss2D(name='', pos=(),special_str=None,  **kwargs):
    '''
    saveGauss2D(name='./result/gauss/time_' + str(t) + '.dat', strain=strain, stress=stress, fabric=fabric)
    :param name:
    :param pos:
    :param kwargs:
    :return:
    '''
    fout = open(name, 'w')
    for key in kwargs:  # strain, stress, fabric
        try:
            data = kwargs[key].toListOfTuples()
        except:
            data = kwargs[key]
        if len(pos) == 0:
            fout.write('%s ' % key + str(len(data)) + '\n')
            for i in range(len(data)):
                if key == 'vR' or key == 'index_large_error' or key == 'H_0' or key == 'H_1' or key == 'numg_index':
                    fout.write('%s\n' % data[i])
                elif key == 'tangent' or key == 'D':
                    temp = [data[i][0][0][0][0], data[i][0][1][0][0], data[i][1][1][0][0],
                            data[i][0][1][0][1], data[i][0][1][1][1], data[i][1][1][1][1]]
                    fout.write(' '.join('%s' % x for x in temp)+'\n')
                elif key == 'frobeniusNorm' or key == 'iteration' or key=='epsPlastic':
                    fout.write('%.4e\n' % data[i])
                elif key == 'epsPlasticVector' or key == 'H_3F':
                    fout.write(' '.join('%s' % x for x in data[i])+'\n')
                else:
                    # if special_str is None:
                    #     fout.write(' '.join('%s %s' % (x[0], x[1]) for x in data[i]) + '\n')  #
                    # else:
                    #     fout.write(' '.join('%s' % x for x in data[i]) + '\n')  #
                    fout.write(' '.join('%s %s' % (x[0], x[1]) for x in data[i]) + '\n')
        else:
            fout.write('%s ' % key + str(len(pos)) + '\n')
            for i in pos:
                fout.write(' '.join('%s %s' % x for x in data[i]) + '\n')
    fout.close()



def saveGauss3D(name='', pos=(),special_str=None,  **kwargs):
    '''
    saveGauss3D(name='./result/gauss/time_' + str(t) + '.dat', strain=strain, stress=stress, fabric=fabric)
    :param name:
    :param pos:
    :param kwargs:
    :return:
    '''
    fout = open(name, 'w')
    for key in kwargs:  # strain, stress, fabric
        try:
            data = kwargs[key].toListOfTuples()
        except:
            data = kwargs[key]
        if len(pos) == 0:
            fout.write('%s ' % key + str(len(data)) + '\n')
            for i in range(len(data)):
                if key == 'vR' or key == 'index_large_error' or key == 'H_0' or key == 'H_1' or key == 'numg_index':
                    fout.write('%s\n' % data[i])
                elif key == 'tangent' or key == 'D':
                    '''
                        stiffness in Voigt Notion 
                       [[0000 0011 0022 0001 0012 0020],
                        [1100 1111 1122 1101 1112 1120],
                        [2200 2211 2222 2201 2212 2220],
                        [0100 0111 0122 0101 0112 0120],
                        [1200 1211 1222 1201 1212 1220],
                        [2000 2011 2022 2001 2012 2020]]
                    '''
                    temp = [data[i][0][0][0][0], data[i][0][0][0][1], data[i][0][0][1][1], data[i][0][0][1][2], data[i][0][0][2][0],data[i][0][0][2][2],
                            data[i][0][1][0][1], data[i][0][1][1][1], data[i][0][1][1][2], data[i][0][1][2][0], data[i][0][1][2][2],
                            data[i][1][1][1][1], data[i][1][1][1][2], data[i][1][1][2][0], data[i][1][1][2][2],
                            data[i][1][2][1][2], data[i][1][2][2][0], data[i][1][2][2][2],
                            data[i][2][0][2][0], data[i][2][0][2][2],
                            data[i][2][2][2][2],
                            ]

                    fout.write(' '.join('%s' % x for x in temp)+'\n')
                elif key == 'frobeniusNorm' or key == 'iteration' or key == 'epsPlastic':
                    fout.write('%.4e\n' % data[i])
                elif key == 'epsPlasticVector'or key == 'H_3D' or key == 'rd_eps':
                    fout.write(' '.join('%s' % x for x in data[i]) + '\n')
                else:
                    # if special_str is None:
                    #     fout.write(' '.join('%s %s' % (x[0], x[1]) for x in data[i]) + '\n')  #
                    # else:
                    #     fout.write(' '.join('%s' % x for x in data[i]) + '\n')  #
                    # fout.write(' '.join('%s %s %s' % (x[0], x[1], x[2]) for x in data[i]) + '\n')  for 3D
                    # temp = [data[i][0][0], data[i][0][1], data[i][0][2],
                    #         data[i][1][1], data[i][1][2],
                    #         data[i][2][2],
                    #         ]
                    # fout.write(' '.join('%s' % x for x in temp)+'\n')
                    fout.write(' '.join('%s %s %s' % (x[0], x[1], x[2]) for x in data[i]) + '\n')
        else:
            fout.write('%s ' % key + str(len(pos)) + '\n')
            for i in pos:
                fout.write(' '.join('%s %s' % x for x in data[i]) + '\n')
    fout.close()








# def saveGauss3D(name='', pos=(), **kwargs):
#     fout = open(name, 'w')
#     for key in kwargs:
#         data = kwargs[key].toListOfTuples()
#         if len(pos) == 0:
#             fout.write('%s ' % key + str(len(data)) + '\n')
#             for i in range(len(data)):
#                 fout.write(' '.join('%s %s %s' % x for x in data[i]) + '\n')
#         else:
#             fout.write('%s ' % key + str(len(pos)) + '\n')
#             for i in pos:
#                 fout.write(' '.join('%s %s %s' % x for x in data[i]) + '\n')
#     fout.close()




def save_loading(save_path, t: int, iter=None, special_str=None,  **kwargs):
    '''
        t: is the time step
        iter: is the iteration num in implicit mode
        kwargs: are values to be saved
    '''
    dir_name = 'iteration_gauss' if special_str is None else 'added_points'
    if iter == None:
        fname = os.path.join(
            save_path, '%s/time_%d' % (dir_name, t))
    else:
        fname = os.path.join(
            save_path, '%s/time_%d_iter_%d' % (dir_name, t, iter))
    if special_str is not None:
        fname += '_%s.dat' % special_str
    else:
        fname += '.dat'
    saveGauss2D(name=fname,special_str=special_str, **kwargs)
    # saveGauss3D(name=fname, special_str=special_str, **kwargs)