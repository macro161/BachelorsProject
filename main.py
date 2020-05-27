import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
import time
import warnings
import sys
import math
import timeit
from datetime import datetime
import pandas as pd
from numpy import genfromtxt


warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')
NODES_X = int(1000)
NODES_Y = int(1000)
NODES_Z = int(100)
SIZE_X = 1.0
SIZE_Y = 1.0
BOUNDARY_X_A = 700
BOUNDARY_X_B = 200
BOUNDARY_Y_A = 0
BOUNDARY_Y_B = 400
BOUNDARY_Z_A = 500
BOUNDARY_Z_B = 600
NEEDLE_TEMP = 0
ERROR_TARGET = 0.001
MIN_TEMP = 0
MAX_TEMP = 253
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
CORES = 4
X = int(2)
Y = int(2)
k = SIZE_X / NODES_X
h = SIZE_Y / NODES_Y
k2 = (SIZE_X / NODES_X) * (SIZE_X / NODES_X)
h2 = (SIZE_Y / NODES_Y) * (SIZE_Y / NODES_Y)
divideBy = 2 * (k2 + h2)
# log = open(str(sys.argv[1]) + 'log.txt','a')

def logit(text):
    i = 10
    # log.write('\n' + str(datetime.now()) + ': ' + str(text) + '\n')


def initialize_matrix():
    b = np.zeros((NODES_Y, NODES_X), dtype='float64')
    b[0,].fill(BOUNDARY_X_A)
    b[NODES_Y - 1,].fill(BOUNDARY_X_B)
    b[:, 0].fill(BOUNDARY_Y_A)
    b[:, NODES_X - 1].fill(BOUNDARY_Y_B)
    return b


def initialize_matrix_3D():
    b = np.zeros((NODES_Y, NODES_X, NODES_Z), dtype='float64')

    b[0, 0:NODES_X, ].fill(BOUNDARY_X_A)
    b[NODES_X - 1, 0:NODES_X, ].fill(BOUNDARY_X_B)
    b[0:NODES_X, 0, ].fill(BOUNDARY_Y_A)
    b[0:NODES_X, NODES_X - 1, ].fill(BOUNDARY_Y_B)
    b[0:NODES_X, 0:NODES_X, 0].fill(BOUNDARY_Z_A)
    b[0:NODES_X, 0:NODES_X, NODES_X - 1].fill(BOUNDARY_Z_B)

    run = True
    a = b.copy()
    while run:
        b[1:NODES_Z - 1, 1:NODES_Y - 1, 1:NODES_X - 1] = np.divide(np.add(
            np.add(
                np.add(b[0:NODES_Z - 2, 1:NODES_Y - 1, 1:NODES_X - 1], b[2:NODES_Z, 1:NODES_Y - 1, 1:NODES_X - 1]),
                np.add(b[1:NODES_Z - 1, 1:NODES_Y - 1, 0:NODES_X - 2], b[1:NODES_Z - 1, 1:NODES_Y - 1, 2:NODES_X])),
            np.add(b[1:NODES_Z - 1, 0:NODES_Y - 2, 1:NODES_X - 1], b[1:NODES_Z - 1, 2:NODES_Y, 1:NODES_X - 1])), 6)
        # logit(str(np.amax(np.multiply(np.divide(np.subtract(b[1:-1, 1:-1, 1:-1], a[1:-1, 1:-1, 1:-1]), b[1:-1, 1:-1, 1:-1]), 100))))
        print(np.amax(np.multiply(np.divide(np.subtract(b[1:-1, 1:-1, 1:-1], a[1:-1, 1:-1, 1:-1]), b[1:-1, 1:-1, 1:-1]), 100)))
        if np.amax(np.multiply(np.divide(np.subtract(b[1:-1, 1:-1, 1:-1], a[1:-1, 1:-1, 1:-1]), b[1:-1, 1:-1, 1:-1]), 100)) < ERROR_TARGET:
            run = False
        a = b.copy()

    # display_3D(b[1:-1, 1:-1, 1:-1])
    # display_color_map(b[30])


def calculate_node_value_gs(x, y, matrix):
    return (matrix[x - 1, y] + matrix[x, y + 1] + matrix[x + 1, y] + matrix[x, y - 1]) / 4


def calculate_error_unoptimized(a, b):
    error = np.zeros((NODES_Y, NODES_X), dtype='float64')
    for i in range(NODES_X - 2):
        for j in range(NODES_X - 2):
            error[i, j] = ((a[i + 1, j + 1] - b[i + 1, j + 1]) / a[i + 1, j + 1]) * 100.0
    if np.amax(error) < ERROR_TARGET:
        return True
    return False


def calculate_error_optimized(a, b):
    return np.amax(((a[1:-1, 1:-1] - b[1:-1, 1:-1]) / a[1:-1, 1:-1]) * 100) < ERROR_TARGET


def calculate_error_super_optimized(a, b):
    return np.amax(np.multiply(np.divide(np.subtract(a[1:-1, 1:-1], b[1:-1, 1:-1]), a[1:-1, 1:-1]), 100)) < ERROR_TARGET


def gauss_seidel_method_unoptimized(matrix):
    model_size = matrix.shape
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()
    while True:
        for i in range(model_size[0] - 2):
            for j in range(model_size[1] - 2):
                next_matrix[i + 1, j + 1] = calculate_node_value_gs(i + 1, j + 1, prev_matrix)
        if calculate_error_unoptimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()


def gauss_seidel_method_small_optimizations(matrix):
    model_size = np.shape(matrix)
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()
    while True:
        for i in range(model_size[0] - 2):
            next_matrix[i + 1, 1:model_size[0] - 1] = (next_matrix[i, 1:model_size[0] - 1] + next_matrix[i + 1,0:model_size[0] - 2] + next_matrix[i + 1, 2:model_size[0]] + next_matrix[i + 2,1:model_size[0] - 1]) / 4
        if calculate_error_optimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()


def gauss_seidel_method_optimized(matrix):
    model_size = np.shape(matrix)
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()
    i = 0
    while i < 1000:
        next_matrix[1:model_size[1] - 1, 1:model_size[0] - 1] = (next_matrix[0:model_size[1] - 2,1:model_size[0] - 1] + next_matrix[1:model_size[1] - 1,0:model_size[0] - 2] + next_matrix[1:model_size[1] - 1,2:model_size[0]] + next_matrix[2:model_size[1],1:model_size[0] - 1]) / 4
        if calculate_error_optimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()
        i = i + 1


def gauss_seidel_method_super_optimized(matrix):
    model_size = np.shape(matrix)
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()
    while True:
        next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
            np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                   next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
            np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                   next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
        if calculate_error_super_optimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()


def forceAspect(ax,aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def display_color_map(matrix):
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    plt.imshow(matrix, cmap='rainbow', interpolation='nearest')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.clim(MIN_TEMP, MAX_TEMP)
    plt.savefig("demo.png")
    plt.show()


def mpi_1x1():
    if rank == 0:
        one_y = int(NODES_Y / 2)
        modelMatrixMpi1x1 = initialize_matrix()
        comm.send(modelMatrixMpi1x1[0:one_y + 1, 0:NODES_X], dest=1, tag=1001)
        comm.send(modelMatrixMpi1x1[one_y - 1:NODES_Y + 1, 0:NODES_X], dest=2, tag=1002)  # Base matrix to node 1
        tic = datetime.now()
        rank_one = comm.recv(source=1, tag=1210)
        rank_two = comm.recv(source=2, tag=1220)
        rank_one = np.delete(rank_one, np.shape(rank_one)[0] - 1, axis=0)
        rank_two = np.delete(rank_two, 0, axis=0)
        toc = datetime.now()
        print(toc - tic)
        big = np.vstack((rank_one, rank_two))
        display_color_map(big)

    if rank == 1:
        matrix = comm.recv(source=0, tag=1001)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            comm.send(next_matrix[model_size[0] - 2, 0:NODES_X], dest=2, tag=1112)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                comm.send(next_matrix, dest=0, tag=1210)
                while True:
                    comm.send(next_matrix[model_size[0] - 2, 0:NODES_X], dest=2, tag=1112)
            next_matrix[model_size[0] - 1, 0:NODES_X] = comm.recv(source=2, tag=1121)
            prev_matrix = next_matrix.copy()

    if rank == 2:
        matrix = comm.recv(source=0, tag=1002)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            comm.send(next_matrix[1, 0:NODES_X], dest=1, tag=1121)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                comm.send(next_matrix, dest=0, tag=1220)
                while True:
                    comm.send(next_matrix[1, 0:NODES_X], dest=1, tag=1121)
            next_matrix[0, 0:NODES_X] = comm.recv(source=1, tag=1112)
            prev_matrix = next_matrix.copy()


def mpi_1xn():
    if rank == 0:
        matrix = initialize_matrix()
        rows = int(NODES_Y / CORES)
        tic = datetime.now()
        for i in range(CORES):
            if i != 0:
                y_start = int(i * rows) - 1
            else:
                y_start = int(i * rows)

            if i == CORES - 1:
                y_end = int(rows * (i + 1))
            else:
                y_end = int(rows * (i + 1) + 1)
            comm.send(matrix[y_start:y_end, 0:NODES_X], dest=i + 1, tag=i + 1)

        data1 = comm.recv(source=1, tag=1)

        print(1)

        data2 = comm.recv(source=2, tag=2)
        print(2)

        data3 = comm.recv(source=3, tag=3)
        print(3)

        data4 = comm.recv(source=4, tag=4)
        print(4)
        toc = datetime.now()
        print toc - tic
        print('JOB DONE')
        full_data = np.vstack((data1[0:np.shape(data1)[0] - 1, 0:NODES_X], data2[1:np.shape(data1)[0], 0:NODES_X]))
        full_data = np.vstack((full_data, data3[1:np.shape(data1)[0], 0:NODES_X]))
        full_data = np.vstack((full_data, data4[1:np.shape(data1)[0], 0:NODES_X]))

        display_color_map(full_data)
    if rank == 1:
        matrix = comm.recv(source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            sendingThis = next_matrix[model_size[0] - 2, 0:NODES_X]
            comm.send(sendingThis, dest=2, tag=1212)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                req = comm.isend(next_matrix, dest=0, tag=rank)

                while True:
                    comm.send(next_matrix[model_size[0] - 2, 0:NODES_X], dest=2, tag=1212)
                req.wait()
            next_matrix[model_size[0] - 1, 0:NODES_X] = comm.recv(source=2, tag=1121)
            prev_matrix = next_matrix.copy()
    if rank == CORES:
        send_code = 1100 + (rank * 10) + (rank - 1)
        recv_code = 1200 + ((rank - 1) * 10) + rank
        matrix = comm.recv(source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            comm.send(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                req = comm.isend(next_matrix, dest=0, tag=rank)

                while True:
                    comm.send(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
                req.wait()
            next_matrix[0, 0:NODES_X] = comm.recv(source=rank - 1, tag=recv_code)
            prev_matrix = next_matrix.copy()
    if (rank > 1) and (rank < CORES):
        matrix = comm.recv(source=0, tag=rank)
        send_one_code = 1100 + (rank * 10) + (rank - 1)
        send_two_code = 1200 + (rank * 10) + (rank + 1)
        recv_one_code = 1200 + ((rank - 1) * 10) + rank
        recv_two_code = 1100 + ((rank + 1) * 10) + rank
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            # print(rank,'end')
            # print('\n',rank,'calculated',next_matrix,'\n')
            # print('\n',rank,' sending ', next_matrix[1, 0:NODES_X], ' to ',rank-1,send_one_code,'\n')
            # print('\n',rank,' sending ', next_matrix[model_size[0] - 2, 0:NODES_X], ' to ',rank+1,send_two_code,'\n')
            comm.send(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
            comm.send(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
            # print(rank,'sent')
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                # print('\n', rank, 'finished with', next_matrix, '\n')
                req = comm.isend(next_matrix, dest=0, tag=rank)

                while True:
                    comm.send(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
                    comm.send(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
                req.wait()
            next_matrix[0, 0:NODES_X] = comm.recv(source=rank - 1, tag=recv_one_code)
            next_matrix[model_size[0] - 1, 0:NODES_X] = comm.recv(source=rank + 1, tag=recv_two_code)
            # print('\n', 'receving', rank, next_matrix[0, 0:NODES_X], rank-1, recv_one_code, '\n')
            # print('\n', 'receving', rank, next_matrix[model_size[0] - 1, 0:NODES_X], rank+1, recv_two_code, '\n')
            # print('\n',rank,next_matrix,'after recv','\n')
            prev_matrix = next_matrix.copy()


def mpi_1xn_new():
    if rank == 0:
        matrix = initialize_matrix()
        rows = int(NODES_Y / CORES)
        tic = datetime.now()
        for i in range(CORES):
            if i != 0:
                y_start = int(i * rows) - 1
            else:
                y_start = int(i * rows)

            if i == CORES - 1:
                y_end = int(rows * (i + 1))
            else:
                y_end = int(rows * (i + 1) + 1)
            comm.Send(matrix[y_start:y_end, 0:NODES_X], dest=i + 1, tag=i + 1)

        full = np.zeros((1, NODES_X), dtype='float64')
        for j in range(CORES):
            # logit(str(j))
            data1 = np.empty([(NODES_Y/CORES)+2,NODES_X])
            reqo = comm.Irecv(data1,source=j+1, tag=j+1)
            reqo.Wait()
            full = np.vstack((full,data1))
        toc = datetime.now()
        print toc - tic
        logit(toc - tic)
        logit('JOB DONE')
        log.close()
        # logit(str(full))
        # display_color_map(full)
        MPI.COMM_WORLD.Abort()

    if rank == 1:
        matrix = np.empty([(NODES_Y/CORES)+1,NODES_X])
        comm.Recv(matrix,source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            sendingThis = next_matrix[model_size[0] - 2, 0:NODES_X]
            reqs = comm.Isend(sendingThis, dest=2, tag=1212)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    rrr = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=2, tag=1212)
                    nmatrix = np.empty([1,NODES_X])
                    reqr = comm.Irecv(nmatrix,source=2, tag=1121)
                    reqr.Wait()
            nmatrix = np.empty([1,NODES_X])
            reqr = comm.Irecv(nmatrix,source=2, tag=1121)
            next_matrix
            reqr.Wait()
            next_matrix[model_size[0] - 1, 0:NODES_X] = nmatrix
            prev_matrix = next_matrix.copy()
    if rank == CORES:
        send_code = 1100 + (rank * 10) + (rank - 1)
        recv_code = 1200 + ((rank - 1) * 10) + rank
        matrix = np.empty([(NODES_Y/CORES)+1,NODES_X])
        comm.Recv(matrix, source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            reqs = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    reqss = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
                    nmatrix = np.empty([1,NODES_X])
                    reqn = comm.Irecv(nmatrix,source=rank - 1, tag=recv_code)
                    reqn.Wait()
                req.wait()
            nmatrix = np.empty([1,NODES_X])
            reqn = comm.Irecv(next_matrix[0, 0:NODES_X],source=rank - 1, tag=recv_code)
            reqn.Wait()
            prev_matrix = next_matrix.copy()
    if (rank > 1) and (rank < CORES):
        matrix = np.empty([(NODES_Y/CORES)+2,NODES_X])
        comm.Recv(matrix,source=0, tag=rank)
        send_one_code = 1100 + (rank * 10) + (rank - 1)
        send_two_code = 1200 + (rank * 10) + (rank + 1)
        recv_one_code = 1200 + ((rank - 1) * 10) + rank
        recv_two_code = 1100 + ((rank + 1) * 10) + rank
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            reqso = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
            reqst = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    reqo = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
                    reqt = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
                    reqn = comm.Irecv(next_matrix[0, 0:NODES_X], source=rank - 1, tag=recv_one_code)
                    reqm = comm.Irecv(next_matrix[model_size[0] - 1, 0:NODES_X], source=rank + 1, tag=recv_two_code)
                    reqn.Wait()
                    reqm.Wait()
                req.Wait()
            reqn = comm.Irecv(next_matrix[0, 0:NODES_X], source=rank - 1, tag=recv_one_code)
            reqm = comm.Irecv(next_matrix[model_size[0] - 1, 0:NODES_X], source=rank + 1, tag=recv_two_code)
            reqn.Wait()
            reqm.Wait()
            prev_matrix = next_matrix.copy()


def mpi_1xn_new_lobotomy():
    if rank == 0:
        matrix = initialize_matrix()
        rows = int(NODES_Y / CORES)
        tic = datetime.now()
        for i in range(CORES):
            if i != 0:
                y_start = int(i * rows) - 1
            else:
                y_start = int(i * rows)

            if i == CORES - 1:
                y_end = int(rows * (i + 1))
            else:
                y_end = int(rows * (i + 1) + 1)
            comm.Send(matrix[y_start:y_end, 0:NODES_X], dest=i + 1, tag=i + 1)

        # full = np.zeros((1, 1), dtype='float64')
        # for j in range(CORES):
        #     # logit(str(j))
        #     data1 = np.empty((1,1),dtype='float64')
        #     reqo = comm.Irecv(data1,source=j+1, tag=j+1)
        #     reqo.Wait()
            # full = np.vstack((full,data1))
        # toc = datetime.now()
        # print toc - tic
        # logit(toc - tic)
        # logit('JOB DONE')
        # log.close()
        # logit(str(full))
        # display_color_map(full)
        # MPI.COMM_WORLD.Abort()

    if rank == 1:
        matrix = np.empty([(NODES_Y/CORES)+1,NODES_X])
        comm.Recv(matrix,source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        count = 1
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            sendingThis = next_matrix[model_size[0] - 2, 0:NODES_X]
            reqs = comm.Isend(sendingThis, dest=2, tag=1212)
            count = count + 1
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                print rank, count
                # res = np.zeros()
                # req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    rrr = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=2, tag=1212)
                    nmatrix = np.empty([1,NODES_X])
                    reqr = comm.Irecv(nmatrix,source=2, tag=1121)
                    reqr.Wait()
            nmatrix = np.empty([1,NODES_X])
            reqr = comm.Irecv(nmatrix,source=2, tag=1121)
            next_matrix
            reqr.Wait()
            next_matrix[model_size[0] - 1, 0:NODES_X] = nmatrix
            prev_matrix = next_matrix.copy()
    if rank == CORES:
        send_code = 1100 + (rank * 10) + (rank - 1)
        recv_code = 1200 + ((rank - 1) * 10) + rank
        matrix = np.empty([(NODES_Y/CORES)+1,NODES_X])
        comm.Recv(matrix, source=0, tag=rank)
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        count = 1
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            reqs = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
            count = count + 1
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                print rank, count 
                # req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    reqss = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_code)
                    nmatrix = np.empty([1,NODES_X])
                    reqn = comm.Irecv(nmatrix,source=rank - 1, tag=recv_code)
                    reqn.Wait()
                req.wait()
            nmatrix = np.empty([1,NODES_X])
            reqn = comm.Irecv(next_matrix[0, 0:NODES_X],source=rank - 1, tag=recv_code)
            reqn.Wait()
            prev_matrix = next_matrix.copy()
    if (rank > 1) and (rank < CORES):
        matrix = np.empty([(NODES_Y/CORES)+2,NODES_X])
        comm.Recv(matrix,source=0, tag=rank)
        send_one_code = 1100 + (rank * 10) + (rank - 1)
        send_two_code = 1200 + (rank * 10) + (rank + 1)
        recv_one_code = 1200 + ((rank - 1) * 10) + rank
        recv_two_code = 1100 + ((rank + 1) * 10) + rank
        model_size = np.shape(matrix)
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        run = True
        count = 1
        while run:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            reqso = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
            reqst = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
            count = count + 1
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                print rank, count
                # req = comm.Isend(next_matrix, dest=0, tag=rank)
                while True:
                    reqo = comm.Isend(next_matrix[1, 0:NODES_X], dest=rank - 1, tag=send_one_code)
                    reqt = comm.Isend(next_matrix[model_size[0] - 2, 0:NODES_X], dest=rank + 1, tag=send_two_code)
                    reqn = comm.Irecv(next_matrix[0, 0:NODES_X], source=rank - 1, tag=recv_one_code)
                    reqm = comm.Irecv(next_matrix[model_size[0] - 1, 0:NODES_X], source=rank + 1, tag=recv_two_code)
                    reqn.Wait()
                    reqm.Wait()
                req.Wait()
            reqn = comm.Irecv(next_matrix[0, 0:NODES_X], source=rank - 1, tag=recv_one_code)
            reqm = comm.Irecv(next_matrix[model_size[0] - 1, 0:NODES_X], source=rank + 1, tag=recv_two_code)
            reqn.Wait()
            reqm.Wait()
            prev_matrix = next_matrix.copy()


def gauss_seidel_method_3D():
    initialize_matrix_3D()


def display_3D(matrix):
    z, x, y = matrix.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cube = ax.scatter(x, y, -z, zdir='z', c=matrix[z, y, x], cmap=plt.cm.rainbow)
    cbar = fig.colorbar(cube, shrink=0.6, aspect=5)
    plt.savefig("demo.png")
    

def mpi_nxn():
    if rank == 0:
        matrix = initialize_matrix()
        # matrix = np.array([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.],
        #                    [13,14,15,16,17,18,19,20,21,22,23,24],
        #                    [25,26,27,28,29,30,31,32,33,34,35,36],
        #                    [37,38,39,40,41,42,43,44,45,46,47,48],
        #                    [49,50,51,52,53,54,55,56,57,58,59,60],
        #                    [61,62,63,64,65,66,67,68,69,70,71,72],
        #                    [73,74,75,76,77,78,79,80,81,82,83,84],
        #                    [85,86,87,88,89,90,91,92,93,94,95,96],
        #                    [97,98,99,100,101,102,103,104,105,106,107,108],
        #                    [109,110,111,112,113,114,115,116,117,118,119,120],
        #                    [121,122,123,124,125,126,127,128,129,130,131,132],
        #                    [133,134,135,136,137,138,139,140,141,142,143,144],
        #                    [145,146,147,148,149,150,151,152,153,154,155,156],
        #                    [157,158,159,160,161,162,163,164,165,166,167,168],
        #                    [169,170,171,172,173,174,175,176,177,178,179,180]])
        # cores_size = int(math.sqrt(CORES))
        matrix_size = np.shape(matrix)
        # full = np.zeros((1,12),dtype='float64')

        tic = datetime.now()
        h = 1
        for i in range(Y):
            for j in range(X):
                y_start = int(i * matrix_size[0]/Y)
                y_end = int(((i + 1) * matrix_size[0] / Y - 1) + 1)
                x_start = int(j * matrix_size[1]/X)
                x_end = int(((j + 1) * matrix_size[1] / X - 1) + 1)
                # send_matrix = np.array(matrix[y_start:y_end,x_start:x_end])
                # print(send_matrix)
                if i == 0 and j == 0:
                    dest1 = -1
                    dest2 = X + 1
                    dest3 = -1
                    dest4 = 2
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start:y_end + 1,x_start:x_end + 1])[0],np.shape(matrix[y_start:y_end + 1,x_start:x_end + 1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:(y_end + 1),x_start:(x_end + 1)]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == 0 and j == X - 1:
                    dest3 = X - 1
                    dest2 = X + X
                    dest1 = -1
                    dest4 = -1
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start:y_end+1,x_start-1:x_end])[0],np.shape(matrix[y_start:y_end+1,x_start-1:x_end])[1]])
                    # print destinations
                    # print 'cancer', destinations, str(((i * (Y-1)) + j + 1)), str((((i * (Y-1)) + j + 1)+1000))
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:y_end+1,x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and j == 0:
                    dest1 = ((Y - 1) * X) - (X - 1)
                    dest4 = (X * (Y - 1)) + 2
                    dest2 = -1
                    dest3 = -1
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start-1:y_end, x_start:x_end+1])[0],np.shape(matrix[y_start-1:y_end, x_start:x_end+1])[1]])
                    # print destinations
                    # print 'cancer', destinations, str(((i * (Y-1)) + j + 1)), str((((i * (Y-1)) + j + 1)+1000))
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and j == X - 1:
                    dest3 = (X * Y) - 1
                    dest1 = X * (Y - 1)
                    dest2 = -1
                    dest4 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end, x_start-1:x_end])[0],np.shape(matrix[y_start-1:y_end, x_start-1:x_end])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and 0 < j < X - 1:
                    dest3 = (X * (Y - 1)) + j
                    dest4 = (X * (Y - 1)) + j + 2
                    dest1 = (X * (Y - 2)) + j + 1
                    dest2 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end, x_start-1:x_end+1])[0],np.shape(matrix[y_start-1:y_end, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == X - 1:
                    dest1 = i * X
                    dest2 = (i + 2) * X
                    dest3 = ((i + 1) * X) - 1
                    dest4 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end])[0],np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == 0 and 0 < j < X - 1:
                    dest3 = j
                    dest4 = j + 2
                    dest2 = j + 1 + X
                    dest1 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start:y_end+1, x_start-1:x_end+1])[0],np.shape(matrix[y_start:y_end+1, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:y_end+1, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == 0:
                    dest4 = (i * X) + 2
                    dest2 = ((i + 1) * X) + 1
                    dest1 = ((i - 1) * X) + 1
                    dest3 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start:x_end+1])[0],np.shape(matrix[y_start-1:y_end+1, x_start:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and 0 < j < X - 1:
                    dest1 = ((i - 1) * X) + j + 1
                    dest3 = (i * X) + j
                    dest4 = (i * X) + j + 2
                    dest2 = ((i + 1) * X) + j + 1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end+1])[0],np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
        
        h = 1
        full = np.zeros((1,NODES_X+1), dtype='float64')
        for i in range(Y):
            partial = np.empty((NODES_Y/Y,1),dtype='float64')
            for j in range(X):
                if i == 0 and j == 0:
                    receiving = np.empty((NODES_Y/Y+1,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'res', receiving[0:3,0:3]
                    partial = np.hstack((partial,receiving[0:NODES_Y/Y,0:NODES_X/X]))
                    # print 'part', partial
                    h = h + 1
                    continue
                if i == 0 and j == X - 1:
                    receiving = np.zeros((NODES_Y/Y+1,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    partial = np.hstack((partial,receiving[0:NODES_Y/Y,1:NODES_X/X+1]))
                    # print 'part', partial
                    # print 'res', receiving[0:3,1:3]
                    h = h + 1
                    continue
                if i == Y - 1 and j == 0:
                    receiving = np.zeros((NODES_Y/Y+1,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,0:NODES_X/X]))
                    # print 'part', receiving[1:4,0:4]
                    # print 'res', receiving
                    h = h + 1
                    continue
                if i == Y - 1 and j == X - 1:
                    receiving = np.zeros((NODES_Y/Y+1,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if i == Y - 1 and 0 < j < X - 1:
                    receiving = np.zeros((NODES_Y/Y+1,NODES_X/X+2),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    # print 'res', receiving[1:4,1:4]
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == X - 1:
                    receiving = np.zeros((NODES_Y/Y+2,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if i == 0 and 0 < j < X - 1:
                    receiving = np.zeros((NODES_Y/Y+1,NODES_X/X+2),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'partial', partial
                    # print 'res', receiving[0:3,1:4]
                    partial = np.hstack((partial,receiving[0:NODES_Y/Y,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == 0:
                    receiving = np.zeros((NODES_Y/Y+2,NODES_X/X+1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    # print 'res', receiving
                    # print 'part', receiving[1:4,0:3]
                    # print 'rank', (i * (Y-1)) + j + 1
                    req.Wait()
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,0:NODES_X/X]))
                    h = h + 1
                    continue
                    
                if 0 < i < Y - 1 and 0 < j < X -1:
                    receiving = np.zeros((NODES_Y/Y+2,NODES_X/X+2),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'res', receiving[1:3,1:3]
                    partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                # print '\n', (i * (Y-1)) + j + 1,'got it', receiving, '\n'
            # print 'full part', partial

            full = np.vstack((full,partial))
            # print 'full', full
                # print('\n','sita rodys kai actaully kazkas veiks',rank,'\n')
        # print full[1:13,]
        toc = datetime.now()
        print toc - tic
        logit(toc-tic)
        log.close()
        # display_color_map(full[1:NODES_Y+1,1:NODES_X+3])
        MPI.COMM_WORLD.Abort()
    else:
        # print rank
        # logit(str(rank))
        destinations = np.zeros((1,6), dtype='int64')
        req = comm.Irecv(destinations, source=0, tag=rank+1000)
        req.Wait()
        matrix = np.zeros((int(destinations[0,4]),int(destinations[0,5])),dtype='float64')
        reqq = comm.Irecv(matrix, source=0, tag=rank+2000)
        reqq.Wait()
        model_size = np.shape(matrix) 
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        while True:
            next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                       next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            if destinations[0,0] != -1:
                sending = next_matrix[1,0:model_size[1]]
                req = comm.Isend(sending,dest=destinations[0,0], tag=destinations[0,0]) 
                req.Wait()
            if destinations[0,1] != -1:
                sending = next_matrix[model_size[0]-2,0:model_size[1]]
                req = comm.Isend(sending,dest=destinations[0,1], tag=destinations[0,1]) 
                req.Wait()
            if destinations[0,2] != -1:
                sending = next_matrix[0:model_size[0],1]
                req = comm.Isend(np.array(sending),dest=destinations[0,2], tag=destinations[0,2]) 
                req.Wait()
            if destinations[0,3] != -1:
                sending = next_matrix[0:model_size[0],model_size[1]-2]
                req = comm.Isend(np.array(sending),dest=destinations[0,3], tag=destinations[0,3])
                req.Wait()

            if calculate_error_super_optimized(next_matrix, prev_matrix):
                logit(str(rank))
                comm.Isend(next_matrix,dest=0,tag=rank)
                while True:
                    if destinations[0,0] != -1:
                        sending = next_matrix[1,0:model_size[1]]
                        req = comm.Isend(np.array(sending),dest=destinations[0,0], tag=destinations[0,0])
                        req.Wait()
                    if destinations[0,1] != -1:
                        sending = next_matrix[model_size[0]-2,0:model_size[1]]
                        req = comm.Isend(np.array(sending),dest=destinations[0,1], tag=destinations[0,1])
                        req.Wait() 
                    if destinations[0,2] != -1:
                        sending = next_matrix[0:model_size[0],1]
                        req = comm.Isend(np.array(sending),dest=destinations[0,2], tag=destinations[0,2]) 
                        req.Wait()
                    if destinations[0,3] != -1:
                        sending = next_matrix[0:model_size[0],model_size[1]-2]
                        req = comm.Isend(np.array(sending),dest=destinations[0,3], tag=destinations[0,3])
                        req.Wait()
            
            if destinations[0,0] != -1:
                receiving = np.zeros((1,model_size[1]),dtype='float64')
                reqqq = comm.Irecv(receiving,source=destinations[0,0], tag=rank)
                reqqq.Wait()'
                next_matrix[0,0:model_size[1]] = receiving
            if destinations[0,1] != -1:
                receiving = np.zeros((1,model_size[1]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,1], tag=rank)
                reqqq.Wait()
                next_matrix[model_size[0]-1,0:model_size[1]] = receiving
            if destinations[0,2] != -1:
                receiving = np.zeros((1,model_size[0]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,2], tag=rank)
                reqqq.Wait()
                next_matrix[0:model_size[0],0] = receiving
            if destinations[0,3] != -1:
                receiving = np.zeros((1,model_size[0]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,3], tag=rank)
                reqqq.Wait()
                next_matrix[0:model_size[0],model_size[1]-1] = receiving

            prev_matrix = next_matrix.copy()
                

def mpi_nxn_new():
    if rank == 0:
        matrix = initialize_matrix()
        # matrix = np.array([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.],
        #                    [13,14,15,16,17,18,19,20,21,22,23,24],
        #                    [25,26,27,28,29,30,31,32,33,34,35,36],
        #                    [37,38,39,40,41,42,43,44,45,46,47,48],
        #                    [49,50,51,52,53,54,55,56,57,58,59,60],
        #                    [61,62,63,64,65,66,67,68,69,70,71,72],
        #                    [73,74,75,76,77,78,79,80,81,82,83,84],
        #                    [85,86,87,88,89,90,91,92,93,94,95,96],
        #                    [97,98,99,100,101,102,103,104,105,106,107,108],
        #                    [109,110,111,112,113,114,115,116,117,118,119,120],
        #                    [121,122,123,124,125,126,127,128,129,130,131,132],
        #                    [133,134,135,136,137,138,139,140,141,142,143,144],
        #                    [145,146,147,148,149,150,151,152,153,154,155,156],
        #                    [157,158,159,160,161,162,163,164,165,166,167,168],
        #                    [169,170,171,172,173,174,175,176,177,178,179,180]])
        # cores_size = int(math.sqrt(CORES))
        matrix_size = np.shape(matrix)
        # full = np.zeros((1,12),dtype='float64')

        tic = datetime.now()
        h = 1
        for i in range(Y):
            for j in range(X):
                y_start = int(i * matrix_size[0]/Y)
                y_end = int(((i + 1) * matrix_size[0] / Y - 1) + 1)
                x_start = int(j * matrix_size[1]/X)
                x_end = int(((j + 1) * matrix_size[1] / X - 1) + 1)
                # send_matrix = np.array(matrix[y_start:y_end,x_start:x_end])
                # print(send_matrix)
                if i == 0 and j == 0:
                    dest1 = -1
                    dest2 = X + 1
                    dest3 = -1
                    dest4 = 2
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start:y_end + 1,x_start:x_end + 1])[0],np.shape(matrix[y_start:y_end + 1,x_start:x_end + 1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:(y_end + 1),x_start:(x_end + 1)]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == 0 and j == X - 1:
                    dest3 = X - 1
                    dest2 = X + X
                    dest1 = -1
                    dest4 = -1
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start:y_end+1,x_start-1:x_end])[0],np.shape(matrix[y_start:y_end+1,x_start-1:x_end])[1]])
                    # print destinations
                    # print 'cancer', destinations, str(((i * (Y-1)) + j + 1)), str((((i * (Y-1)) + j + 1)+1000))
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:y_end+1,x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and j == 0:
                    dest1 = ((Y - 1) * X) - (X - 1)
                    dest4 = (X * (Y - 1)) + 2
                    dest2 = -1
                    dest3 = -1
                    destinations = np.array([dest1,dest2,dest3,dest4,np.shape(matrix[y_start-1:y_end, x_start:x_end+1])[0],np.shape(matrix[y_start-1:y_end, x_start:x_end+1])[1]])
                    # print destinations
                    # print 'cancer', destinations, str(((i * (Y-1)) + j + 1)), str((((i * (Y-1)) + j + 1)+1000))
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and j == X - 1:
                    dest3 = (X * Y) - 1
                    dest1 = X * (Y - 1)
                    dest2 = -1
                    dest4 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end, x_start-1:x_end])[0],np.shape(matrix[y_start-1:y_end, x_start-1:x_end])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == Y - 1 and 0 < j < X - 1:
                    dest3 = (X * (Y - 1)) + j
                    dest4 = (X * (Y - 1)) + j + 2
                    dest1 = (X * (Y - 2)) + j + 1
                    dest2 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end, x_start-1:x_end+1])[0],np.shape(matrix[y_start-1:y_end, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == X - 1:
                    dest1 = i * X
                    dest2 = (i + 2) * X
                    dest3 = ((i + 1) * X) - 1
                    dest4 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end])[0],np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start-1:x_end]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if i == 0 and 0 < j < X - 1:
                    dest3 = j
                    dest4 = j + 2
                    dest2 = j + 1 + X
                    dest1 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start:y_end+1, x_start-1:x_end+1])[0],np.shape(matrix[y_start:y_end+1, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start:y_end+1, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == 0:
                    dest4 = (i * X) + 2
                    dest2 = ((i + 1) * X) + 1
                    dest1 = ((i - 1) * X) + 1
                    dest3 = -1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start:x_end+1])[0],np.shape(matrix[y_start-1:y_end+1, x_start:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and 0 < j < X - 1:
                    dest1 = ((i - 1) * X) + j + 1
                    dest3 = (i * X) + j
                    dest4 = (i * X) + j + 2
                    dest2 = ((i + 1) * X) + j + 1
                    destinations = np.array([dest1, dest2, dest3, dest4,np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end+1])[0],np.shape(matrix[y_start-1:y_end+1, x_start-1:x_end+1])[1]])
                    # print destinations
                    req = comm.Isend(destinations, dest=h, tag=h+1000)
                    reqq = comm.Isend(np.array(matrix[y_start-1:y_end+1, x_start-1:x_end+1]), dest=h, tag=h+2000)
                    req.Wait()
                    reqq.Wait()
                    h = h + 1
        
        h = 1
        # full = np.zeros((1,NODES_X+1), dtype='float64')
        for i in range(Y):
            # partial = np.empty((NODES_Y/Y,1),dtype='float64')
            for j in range(X):
                if i == 0 and j == 0:
                    receiving = np.empty((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'res', receiving[0:3,0:3]
                    # partial = np.hstack((partial,receiving[0:NODES_Y/Y,0:NODES_X/X]))
                    # print 'part', partial
                    h = h + 1
                    continue
                if i == 0 and j == X - 1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # partial = np.hstack((partial,receiving[0:NODES_Y/Y,1:NODES_X/X+1]))
                    # print 'part', partial
                    # print 'res', receiving[0:3,1:3]
                    h = h + 1
                    continue
                if i == Y - 1 and j == 0:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,0:NODES_X/X]))
                    # print 'part', receiving[1:4,0:4]
                    # print 'res', receiving
                    h = h + 1
                    continue
                if i == Y - 1 and j == X - 1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if i == Y - 1 and 0 < j < X - 1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    # print 'res', receiving[1:4,1:4]
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == X - 1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if i == 0 and 0 < j < X - 1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'partial', partial
                    # print 'res', receiving[0:3,1:4]
                    # partial = np.hstack((partial,receiving[0:NODES_Y/Y,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                if 0 < i < Y - 1 and j == 0:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    # print 'res', receiving
                    # print 'part', receiving[1:4,0:3]
                    # print 'rank', (i * (Y-1)) + j + 1
                    req.Wait()
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,0:NODES_X/X]))
                    h = h + 1
                    continue
                    
                if 0 < i < Y - 1 and 0 < j < X -1:
                    receiving = np.zeros((1,1),dtype='float64')
                    req = comm.Irecv(receiving,source=h,tag=h)
                    req.Wait()
                    # print 'res', receiving[1:3,1:3]
                    # partial = np.hstack((partial,receiving[1:NODES_Y/Y+1,1:NODES_X/X+1]))
                    h = h + 1
                    continue
                # print '\n', (i * (Y-1)) + j + 1,'got it', receiving, '\n'
            # print 'full part', partial

            # full = np.vstack((full,partial))
            # print 'full', full
                # print('\n','sita rodys kai actaully kazkas veiks',rank,'\n')
        # print full[1:13,]
        toc = datetime.now()
        print toc - tic
        # logit(toc-tic)
        # log.close()
        # display_color_map(full[1:NODES_Y+1,1:NODES_X+3])
        MPI.COMM_WORLD.Abort()
    else:
        # print rank
        # logit(str(rank))
        destinations = np.zeros((1,6), dtype='int64')
        req = comm.Irecv(destinations, source=0, tag=rank+1000)
        req.Wait()
        matrix = np.zeros((int(destinations[0,4]),int(destinations[0,5])),dtype='float64')
        reqq = comm.Irecv(matrix, source=0, tag=rank+2000)
        reqq.Wait()
        model_size = np.shape(matrix) 
        prev_matrix = matrix.copy()
        next_matrix = matrix.copy()
        # if rank == 1:
        #     print matrix
        count = 1
        while True:
            for i in range(X * Y):
                next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                    np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                        next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                    np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                        next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
            if destinations[0,0] != -1:
                sending = next_matrix[1,0:model_size[1]]
                req = comm.Isend(sending,dest=destinations[0,0], tag=destinations[0,0]) 
                # if rank == 9:
                #     print('\n',rank,'sending',sending,destinations[0,0], '\n')
                req.Wait()
            if destinations[0,1] != -1:
                sending = next_matrix[model_size[0]-2,0:model_size[1]]
                req = comm.Isend(sending,dest=destinations[0,1], tag=destinations[0,1]) 
                # if rank == 9:
                #     print('\n',rank,'sending',sending,destinations[0,1],'\n')
                req.Wait()
            if destinations[0,2] != -1:
                sending = next_matrix[0:model_size[0],1]
                req = comm.Isend(np.array(sending),dest=destinations[0,2], tag=destinations[0,2]) 
                # if rank == 9:
                #     print('\n',rank,'sending',sending,destinations[0,2],'\n') 
                req.Wait()
            if destinations[0,3] != -1:
                sending = next_matrix[0:model_size[0],model_size[1]-2]
                req = comm.Isend(np.array(sending),dest=destinations[0,3], tag=destinations[0,3])
                # if rank == 9:
                #     print('\n',rank,'sending',sending,destinations[0,3],'\n')
                req.Wait()
            count = count + 1
            if calculate_error_super_optimized(next_matrix, prev_matrix):
                print rank, count
                # print('\n','end', rank,next_matrix,'\n')
                # if rank == 9:
                # print rank, 'end', next_matrix
                logit(str(rank))
                receiving = np.zeros((1,1),dtype='float64')
                comm.Isend(receiving,dest=0,tag=rank)
                while True:
                    # next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
                    # np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                    #     next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
                    # np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                    #     next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
                    if destinations[0,0] != -1:
                        sending = next_matrix[1,0:model_size[1]]
                        req = comm.Isend(sending,dest=destinations[0,0], tag=destinations[0,0]) 
                        # if rank == 9:
                        #     print('\n',rank,'sending',sending,destinations[0,0], '\n')
                        req.Wait()
                    if destinations[0,1] != -1:
                        sending = next_matrix[model_size[0]-2,0:model_size[1]]
                        req = comm.Isend(sending,dest=destinations[0,1], tag=destinations[0,1]) 
                        # if rank == 9:
                        #     print('\n',rank,'sending',sending,destinations[0,1],'\n')
                        req.Wait()
                    if destinations[0,2] != -1:
                        sending = next_matrix[0:model_size[0],1]
                        req = comm.Isend(np.array(sending),dest=destinations[0,2], tag=destinations[0,2]) 
                        # if rank == 9:
                        #     print('\n',rank,'sending',sending,destinations[0,2],'\n') 
                        req.Wait()
                    if destinations[0,3] != -1:
                        sending = next_matrix[0:model_size[0],model_size[1]-2]
                        req = comm.Isend(np.array(sending),dest=destinations[0,3], tag=destinations[0,3])
                        # if rank == 9:
                        #     print('\n',rank,'sending',sending,destinations[0,3],'\n')
                        req.Wait()
                
                    if destinations[0,0] != -1:
                        receiving = np.zeros((1,model_size[1]),dtype='float64')
                        reqqq = comm.Irecv(receiving,source=destinations[0,0], tag=rank)
                        reqqq.Wait()
                        # print '\n',rank,'reviecing',receiving, destinations[0,0],'\n'
                        # if rank == 6:
                        #     print '\n',rank, 'before', destinations[0,0], next_matrix, '\n'
                        next_matrix[0,0:model_size[1]] = receiving
                        # if rank == 6:
                        #     print '\n',rank, 'after', destinations[0,0], next_matrix, '\n'
                    if destinations[0,1] != -1:
                        receiving = np.zeros((1,model_size[1]),dtype='float64')
                        reqqq =  comm.Irecv(receiving,source=destinations[0,1], tag=rank)
                        reqqq.Wait()
                        next_matrix[model_size[0]-1,0:model_size[1]] = receiving
                    if destinations[0,2] != -1:
                        receiving = np.zeros((1,model_size[0]),dtype='float64')
                        reqqq =  comm.Irecv(receiving,source=destinations[0,2], tag=rank)
                        reqqq.Wait()
                        next_matrix[0:model_size[0],0] = receiving
                    if destinations[0,3] != -1:
                        receiving = np.zeros((1,model_size[0]),dtype='float64')
                        reqqq =  comm.Irecv(receiving,source=destinations[0,3], tag=rank)
                        reqqq.Wait()
                        next_matrix[0:model_size[0],model_size[1]-1] = receiving

                    prev_matrix = next_matrix.copy()

            if destinations[0,0] != -1:
                receiving = np.zeros((1,model_size[1]),dtype='float64')
                reqqq = comm.Irecv(receiving,source=destinations[0,0], tag=rank)
                reqqq.Wait()
                next_matrix[0,0:model_size[1]] = receiving
            if destinations[0,1] != -1:
                receiving = np.zeros((1,model_size[1]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,1], tag=rank)
                reqqq.Wait()
                next_matrix[model_size[0]-1,0:model_size[1]] = receiving
            if destinations[0,2] != -1:
                receiving = np.zeros((1,model_size[0]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,2], tag=rank)
                reqqq.Wait()
                next_matrix[0:model_size[0],0] = receiving
            if destinations[0,3] != -1:
                receiving = np.zeros((1,model_size[0]),dtype='float64')
                reqqq =  comm.Irecv(receiving,source=destinations[0,3], tag=rank)
                reqqq.Wait()
                next_matrix[0:model_size[0],model_size[1]-1] = receiving

            prev_matrix = next_matrix.copy()


def util_test():
    # matrix = np.zeros((100000000))
    # matrix = np.random.randint(1000,size=10000000)
    matrix = np.random.uniform(low=0.1, high=1000.1, size=(1000000000,))
    # print(matrix)
    if rank == 0:
        print '0'
        tic = datetime.now()
        comm.Send(matrix, dest=1)
        comm.Recv(matrix,source=1)
        toc = datetime.now()
        print toc - tic

    if rank == 1:
        print '1'
        comm.Recv(matrix,source=0)
        comm.Send(matrix,dest=0)


def benchy():
    matrix = initialize_matrix()
    model_size = np.shape(matrix)
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()
    run = True
    i = 0
    while run:
        i = i + 1
        if(i == 1000):
            run = False
        # i = i + 1
        # print(i)
        next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = np.divide(np.add(
            np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],
                   next_matrix[1:model_size[0] - 1, 2:model_size[1]]),
            np.add(next_matrix[2:model_size[0], 1:model_size[1] - 1],
                   next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2])), 4)
        if calculate_error_super_optimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()


def approx_deriv(matrix):
    matrix[0,] = matrix[1,]
    matrix[0:NODES_Y,0] = matrix[0:NODES_Y,1]
    # formula = lambda t: np.absolute(np.sin((np.pi * t)/0.01))
    # result = np.array([formula(xi) for xi in matrix[1,]])
    # matrix[0,] = result
    # h = 0.01
    # result = np.divide(np.subtract(matrix[1, ],matrix[0, ]),h)
    # matrix[0, ] = result


def gauss_seidel_method_super_optimized_final(matrix):
    model_size = np.shape(matrix)
    prev_matrix = matrix.copy()
    next_matrix = matrix.copy()

    # bottom(next_matrix)
    # next_matrix[NODES_Y-1,] = step_function()
    
    while True:
        # approx_deriv(next_matrix)
        # needle(next_matrix)
        next_matrix[1:model_size[0] - 1, 1:model_size[1] - 1] = \
            np.divide(
                np.add(
                    np.multiply(k2,
                                np.add(next_matrix[0:model_size[0] - 2, 1:model_size[1] - 1],next_matrix[2:model_size[0], 1:model_size[1] - 1])),
                    np.multiply(h2,
                                np.add(next_matrix[1:model_size[0] - 1, 2:model_size[1]],next_matrix[1:model_size[0] - 1, 0:model_size[1] - 2]))),divideBy)
        if calculate_error_super_optimized(next_matrix, prev_matrix):
            return next_matrix
        prev_matrix = next_matrix.copy()


def needle(matrix):
    matrix[0,0:50] = NEEDLE_TEMP


def shrink_me():
    matrix = np.array([ [1,2,3,4,5,6,7,8],
                        [5,6,7,8,9,10,11,12],
                        [9,10,11,12,13,14,15,16],
                        [13,14,15,16,15,16,17,18],
                        [1,2,3,4,5,6,7,8],
                        [5,6,7,8,9,10,11,12],
                        [9,10,11,12,13,14,15,16],
                        [13,14,15,16,15,16,17,18]])
    pd.DataFrame(matrix).to_csv("help.csv")


def bottom(matrix):
    matrix[NODES_Y-1,] = sinusoid()


def sinusoid():
    k = 100
    n = 10
    u = 15.0 
    matrix = sin_generator()
    x = 0.0
    result = k * np.sin(matrix * np.pi / 180.)
    print result
    return result


def sin_generator():
    array = np.array([90.])
    hills = 6
    step = (90. * hills) / NODES_X
    start = 90.


    l = 1
    for j in range(hills):
        for i in range(NODES_X/hills+1):
            if (j + 1) % 2 > 0:
                start = start - step
                array = np.append(array, start)
            if (j + 1) % 2 == 0:
                start = start + step
                array = np.append(array, start)
            l = l + 1 
            if l == NODES_X:
                return array
    return array


def step_function():
    matrix = np.array([0.])
    steps = 10
    value = 100.
    l = 1
    for i in range(steps):
        for j in range(NODES_X/steps+1):
            if (i+1) % 2 > 0:
                matrix = np.append(matrix, 0.)
            if (i+1) % 2 == 0:
                matrix = np.append(matrix, value)
            l = l + 1
            if l == NODES_X:
                return matrix
    return matrix
            


# print('UNOPTIMIZED')
# modelMatrix = init
# ialize_matrix()
# tic = time.perf_counter()
# newMatrix = gauss_seidel_method_unoptimized(modelMatrix)
# toc = time.perf_counter()
# print(f'Time: {toc - tic:0.4f}')
# print(newMatrix)
# display_color_map(newMatrix)


# print('SMALL OPTIMIZATION')
# modelMatrixX = initialize_matrix()
# tic = time.perf_counter()
# newMatrixX = gauss_seidel_method_small_optimizations(modelMatrixX)
# toc = time.perf_counter()
# print(f'Time: {toc - tic:0.4f}')
# display_color_map(newMatrixX)


# print('OPTIMIZED')
# modelMatrixXx = initialize_matrix()
# tic = time.perf_counter()
# newMatrixXx = gauss_seidel_method_optimized(modelMatrixXx)
# toc = time.perf_counter()
# print(f'Time: {toc - tic:0.4f}')
# display_color_map(newMatrixXx)


# print('SUPER OPTIMIZED')
# modelMatrixXxx = initialize_matrix()#np. array([[1, 1, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4], [5, 0, 5], [6, 0, 6], [7, 0, 7], [8, 8, 8]])
# tic = datetime.now()
# newMatrixXxx = gauss_seidel_method_super_optimized(modelMatrixXxx)
# toc = datetime.now()
# print toc - tic
# print(newMatrixXxx)
# display_color_map(newMatrixXxx)




# print('MPI ; N cores ; 1xn')
# mpi_1xn()


# print('MPI ; N cores ; 1xn')
# mpi_1xn_new()

print 'MPI ; N cores ; NxN'
mpi_nxn_new()

# print 'Sequential 3D'
# tic = datetime.now()
# gauss_seidel_method_3D()
# toc = datetime.now()
# print toc - tic

# tic = datetime.now()
# benchy()
# toc = datetime.now()
# print toc - tic

# approx_deriv()

# print('SUPER OPTIMIZED FINAL')
# modelMatrixXxx = initialize_matrix()#np. array([[1, 1, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4], [5, 0, 5], [6, 0, 6], [7, 0, 7], [8, 8, 8]])
# tic = datetime.now()
# newMatrixXxx = gauss_seidel_method_super_optimized_final(modelMatrixXxx)
# # pd.DataFrame(newMatrixXxx).to_csv("10000x10000.csv")
# toc = datetime.now()
# print toc - tic
# print(newMatrixXxx)
# display_color_map(newMatrixXxx)

# shrink_me()

# sinusoid()

# sin_generator()

# print('MPI ; N cores ; 1xn')
# mpi_1xn_new_lobotomy()
