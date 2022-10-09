#
# ---------------------------point to point---------------------------------
#
# from mpi4py import MPI
#
#
# comm = MPI.COMM_WORLD
# comm_rank = comm.Get_rank()
# comm_size = comm.Get_size()
# print('rank %d size %d' % (comm_rank, comm_size))
#
# data_send = [comm_rank] * comm_size
# comm.send(data_send, dest=(comm_rank + 1) % comm_size)
# #
# data_recv = comm.recv(source=(comm_rank - 1) % comm_size)
# print("my rank is %d, I received : %s " % (comm_rank, data_recv))

# ----------------------------broadcast-----------------------------------------
# import mpi4py.MPI as MPI
#
# comm = MPI.COMM_WORLD
# comm_rank = comm.Get_rank()
# comm_size = comm.Get_size()
#
# if comm_rank == 0:
#     data = [i for i in range(comm_size)]
#
# data = comm.bcast(data if comm_rank == 0 else None, root=0)
# print("rank %d, got :  %s " % (comm_rank, data))


# ----------------------------scatter-----------------------------------------
import mpi4py.MPI as MPI
import numpy as np

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    data = np.random.rand(comm_size, 3)
    # data = [i for i in range(comm_size)]
    # data = [[1], [2], [3], [4]]
    print("all data by rank %d : " % comm_rank)
    print(data)
else:
    data = None

local_data = comm.scatter(data, root=0)
print("rank %d, got : %s" % (comm_rank, local_data))
