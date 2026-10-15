##################################################
#             Miao Hui-hui, aspirant             #
# Faculty CMC, Lomonosov Moscow State University #
#                hhmiao@cs.msu.ru                #
##################################################


# -*- coding:utf-8 -*-
import numpy as np
import scipy.linalg as scpl
from mpi4py import MPI
import time
import copy
np.seterr(divide = 'ignore', invalid = 'ignore')

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10

if(rank == 0):
    total_start = time.time()


##################################################
##########       Cannon algorithm       ##########
##################################################

def Cannon(matrix_a, matrix_b, index_x, index_y, n_x, n_y, n_local_x, n_local_y):

    # Initialization
    time_tmp_1 = []
    time_tmp_2 = []

    start1 = time.time()
    dest_x = ((index_x - index_y) % n_x) + index_y * n_x
    source_x = ((index_x + index_y) % n_x) + index_y * n_x
    dest_y = index_x + ((index_y - index_x) % n_y) * n_x
    source_y = index_x + ((index_y + index_x) % n_y) * n_x
    end1 = time.time()
    time_tmp_1.append(end1 - start1)

    start2 = time.time()
    if(index_y > 0):
        comm.Sendrecv_replace(matrix_a, dest = dest_x, source = source_x)
    if(index_x > 0):
        comm.Sendrecv_replace(matrix_b, dest = dest_y, source = source_y)
    end2 = time.time()
    time_tmp_2.append(end2 - start2)

    # Shift and multiplication
    start1 = time.time()
    dest_shift_x = ((index_x - 1) % n_x) + index_y * n_x
    source_shift_x = ((index_x + 1) % n_x) + index_y * n_x
    dest_shift_y = index_x + ((index_y - 1) % n_y) * n_x
    source_shift_y = index_x + ((index_y + 1) % n_y) * n_x
    
    matrix_c = np.zeros((n_local_x, n_local_y), dtype = 'complex_')
    matrix_c += np.dot(matrix_a, matrix_b)
    end1 = time.time()
    time_tmp_1.append(end1 - start1)
    for j in range(n_x - 1):
        start2 = time.time()
        comm.Sendrecv_replace(matrix_a, dest = dest_shift_x, source = source_shift_x)
        comm.Sendrecv_replace(matrix_b, dest = dest_shift_y, source = source_shift_y)
        end2 = time.time()
        time_tmp_2.append(end2 - start2)
        start1 = time.time()
        matrix_c += np.dot(matrix_a, matrix_b)
        end1 = time.time()
        time_tmp_1.append(end1 - start1)
    
    return matrix_c, time_tmp_1, time_tmp_2


##################################################
##########        Distributed QME       ##########
##################################################

def QME(rho, rho_copy, local_disses_hor, local_disses_ver, local_influxes_hor, local_influxes_ver, local_replace):

    time_tmp_1 = []
    time_tmp_2 = []

    start1 = time.time()
    for i in range(len(local_disses_hor)):
        for j in range(n_local_y):
            rho[local_disses_hor[i][1]][j] -= rho_copy[local_disses_hor[i][1]][j] * local_disses_hor[i][0]
    for i in range(len(local_disses_ver)):
        for j in range(n_local_x):
            rho[j][local_disses_ver[i][1]] -= rho_copy[j][local_disses_ver[i][1]] * local_disses_ver[i][0]
    for i in range(len(local_influxes_hor)):
        for j in range(n_local_y):
            rho[local_influxes_hor[i][1]][j] -= rho_copy[local_influxes_hor[i][1]][j] * local_influxes_hor[i][0]
    for i in range(len(local_influxes_ver)):
        for j in range(n_local_x):
            rho[j][local_influxes_ver[i][1]] -= rho_copy[j][local_influxes_ver[i][1]] * local_influxes_ver[i][0]
    end1 = time.time()
    time_tmp_1.append(end1 - start1)

    for i in range(len(local_replace)):
        if(local_replace[i][2] == local_replace[i][3]):
            start1 = time.time()
            if(rank == local_replace[i][2]):
                rho[local_replace[i][5]][local_replace[i][5]] += rho_copy[local_replace[i][4]][local_replace[i][4]] * local_replace[i][0]
                rho[local_replace[i][4]][local_replace[i][4]] += rho_copy[local_replace[i][5]][local_replace[i][5]] * local_replace[i][1]
            end1 = time.time()
            time_tmp_1.append(end1 - start1)
        else:
            if(rank == local_replace[i][2]):
                start2 = time.time()
                data_received = comm.sendrecv(rho_copy[local_replace[i][4]][local_replace[i][4]] * local_replace[i][0], dest = local_replace[i][3], source = local_replace[i][3])
                end2 = time.time()
                time_tmp_2.append(end2 - start2)
                start1 = time.time()
                rho[local_replace[i][4]][local_replace[i][4]] += data_received
                end1 = time.time()
                time_tmp_1.append(end1 - start1)
            if(rank == local_replace[i][3]):
                start2 = time.time()
                data_received = comm.sendrecv(rho_copy[local_replace[i][5]][local_replace[i][5]] * local_replace[i][1], dest = local_replace[i][2], source = local_replace[i][2])
                end2 = time.time()
                time_tmp_2.append(end2 - start2)
                start1 = time.time()
                rho[local_replace[i][5]][local_replace[i][5]] += data_received
                end1 = time.time()
                time_tmp_1.append(end1 - start1)

    return rho, time_tmp_1, time_tmp_2

for ii in range(N):

    time1 = 0.
    time2 = 0.
    time3 = 0.
    time3_1 = 0.
    time3_2 = 0.
    time4 = 0.
    time4_1 = 0.
    time4_2 = 0.


    ##################################################
    ##########     GENERATION OF STATES     ##########
    ##################################################

    comm.barrier()
    start = time.time()

    n_atom = 5 # Number of atoms
    list_states_tmp = []
    cnt1 = -1
    for i1 in range(2):
        cnt1 += 1
        cnt2 = -1
        for i2 in range(2):
            cnt2 += 1
            cnt3 = -1
            for i3 in range(2):
                cnt3 += 1
                cnt4 = -1
                for i4 in range(2):
                    cnt4 += 1
                    cnt5 = -1
                    for i5 in range(2):
                        cnt5 += 1
                        cnt6 = -1
                        for i6 in range(2):
                            cnt6 += 1
                            cnt7 = -1
                            for i7 in range(2):
                                cnt7 += 1
                                cnt8 = -1
                                for i8 in range(2):
                                    cnt8 += 1
                                    cnt9 = -1
                                    for i9 in range(2):
                                        cnt9 += 1
                                        cnt10 = -1
                                        for i10 in range(2):
                                            cnt10 += 1
                                            if((cnt1 + cnt2 <= 1) and (cnt3 + cnt4 <= 1) and (cnt5 + cnt6 <= 1) and (cnt7 + cnt8 <= 1) and (cnt9 + cnt10 <= 1)):
                                                list_states_tmp.append([cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7, cnt8, cnt9, cnt10])
    list_states = []
    for i in range(n_atom + 1):
        for j in range(len(list_states_tmp)):
            summa = 0
            for k in range(len(list_states_tmp[0])):
                summa += list_states_tmp[j][k]
            if(summa == (n_atom - i)):
                list_states.append(list_states_tmp[j])

    list_inters = []
    for i in range(len(list_states)):
        for j in range(len(list_states)):
            tmp = 0
            check0 = 0
            check1 = 0
            for k in range(int(len(list_states[0]) / 2)):
                if(list_states[i][2 * k] == list_states[j][2 * k] and list_states[i][2 * k + 1] == list_states[j][2 * k + 1]):
                    tmp += 1
                check0 += (list_states[i][2 * k] + list_states[i][2 * k + 1])
                check1 += (list_states[j][2 * k] + list_states[j][2 * k + 1])
            if(tmp == n_atom - 1 and check0 == check1):
                list_inters.append([i, j])

    list_disses = []
    for i in range(len(list_states)):
        for j in range(len(list_states)):
            tmp = 0
            check = 0
            check0 = 0
            check1 = 0
            for k in range(int(len(list_states[0]) / 2)):
                if(list_states[i][2 * k] == list_states[j][2 * k] and list_states[i][2 * k + 1] == list_states[j][2 * k + 1]):
                    tmp += 1
                else:
                    if(list_states[i][2 * k] != list_states[j][2 * k]):
                        check = 1
                check0 += (list_states[i][2 * k] + list_states[i][2 * k + 1])
                check1 += (list_states[j][2 * k] + list_states[j][2 * k + 1])
            if(tmp == n_atom - 1 and check == 1 and check0 != check1 and i < j):
                list_disses.append([i, j])
    #if(rank == 0):
    #    print(list_states)
    #    print(len(list_states))
    #    print(list_inters)
    #    print(len(list_inters))
    #    print(list_disses)
    #    print(len(list_disses))
    #exit()


    ##################################################
    ##########          PARAMETERS          ##########
    ##################################################

    n_total = len(list_states) # Hamiltonian dimension
    for i in range(16):
        if(n_total % 16 != 0):
            n_total += 1
            state_empty = []
            for j in range(len(list_states[0])):
                state_empty.append(0)
            list_states.append(state_empty)
        else:
            break
    #if(rank == 0):
    #    print(len(list_states))
    #exit()
    n_x = 16 # Number of computing cores along x axis 
    n_y = 16 # Number of computing cores along y axis
    n_core = n_x * n_y # Number of computing cores
    n_local_x = int(n_total / n_x) # Hamiltonian local dimension x
    n_local_y = int(n_total / n_y) # Hamiltonian local dimension y
    #if(rank == 0):
    #    print(n_local_x)
    #    print(n_local_y)
    index_x = rank % n_x
    index_y = rank // n_x
    #print("Rank %d: %d, %d" % (rank, index_x, index_y))

    hbar = 1. # Reduced Planck constant
    w = 10 ** 8 # Photon frequency
    g = 10 ** 6 # Strength
    gamma = 10 ** 6 # Dissipation intensity
    mu = 0.
    t = 10 ** -7 # Period
    iteration = 100 # Interations
    d_tau = t / iteration # Time span
    exponent = 20 # Taylor series

    local_disses_hor = []
    local_disses_ver = []
    local_influxes_hor = []
    local_influxes_ver = []
    local_replace = []
    for i in range(len(list_disses)):
        index1 = list_disses[i][0] // n_local_y
        index2 = list_disses[i][0] % n_local_y
        if(index1 == index_y):
            local_disses_hor.append([0.5 * 1 * gamma * d_tau, index2])
        if(index1 == index_x):
            local_disses_ver.append([0.5 * 1 * gamma * d_tau, index2])

        index1 = list_disses[i][1] // n_local_y
        index2 = list_disses[i][1] % n_local_y
        if(index1 == index_y):
            local_influxes_hor.append([0.5 * 1 * gamma * d_tau * mu, index2])
        if(index1 == index_x):
            local_influxes_ver.append([0.5 * 1 * gamma * d_tau * mu, index2])

        index1 = list_disses[i][0] // n_local_y
        index2 = list_disses[i][1] // n_local_y
        local1 = list_disses[i][0] % n_local_y
        local2 = list_disses[i][1] % n_local_y
        local_replace.append([1 * gamma * d_tau, 1 * gamma * d_tau * mu, index1 + index1 * n_x, index2 + index2 * n_x, local1, local2])

    #if(rank == 1):
    #    print(list_disses)
    #    print(local_replace)


    ##################################################
    ##########         HAMILITONIAN         ##########
    ##################################################

    # Build the Hamiltonian with the help of above groups

    H_m = np.zeros((n_local_x, n_local_y))
    for i in range(len(list_inters)):
        index1 = list_inters[i][0] - index_y * n_local_y
        index2 = list_inters[i][1] - index_x * n_local_x
        if(index1 >= 0 and index1 < n_local_x and index2 >= 0 and index2 < n_local_y):
            H_m[index1][index2] = g
    if(index_x == index_y):
        for i in range(n_local_x):
            cnt = 0
            for j in range(len(list_states[i + n_local_x * index_x])):
                cnt += list_states[i + n_local_x * index_x][j]
            H_m[i][i] = cnt * hbar * w
    #if(rank == 1):
    #    print(H_m)
    #exit()

    end = time.time()
    time1 += end - start
    comm.barrier()

    # Build the initial density matrix
    rho_0 = np.zeros((n_local_x, n_local_y))
    if(rank == 0):
        rho_0[0][0] = 1.
    rho = copy.deepcopy(rho_0)

    if(index_x == index_y):
        rho_ev = []

        # Diagonal of Hamiltonian
        tmp = []
        for i in range(n_local_x):
	        tmp.append(np.abs(rho[i, i]))
        rho_ev.append(np.array(tmp))


    ##################################################
    ##########        Taylor series         ##########
    ##################################################

    comm.barrier()
    time_tmp_1 = []
    time_tmp_2 = []
    start = time.time()

    H_p_backup =  1j * H_m * d_tau
    H_m_backup =  -1j * H_m * d_tau
    if(index_x == index_y):
        exp_H_p = np.eye(n_local_x, dtype = 'complex_')
        exp_H_m = np.eye(n_local_x, dtype = 'complex_')
    else:
        exp_H_p = np.zeros((n_local_x, n_local_y), dtype = 'complex_')
        exp_H_m = np.zeros((n_local_x, n_local_y), dtype = 'complex_')
    exp_H_p += H_p_backup
    exp_H_m += H_m_backup
    factorial = 1
    H_p_tmp1 = copy.deepcopy(H_p_backup)
    H_p_tmp2 = copy.deepcopy(H_p_backup)
    H_m_tmp1 = copy.deepcopy(H_m_backup)
    H_m_tmp2 = copy.deepcopy(H_m_backup)
    for i in range(2, exponent + 1):
        factorial *= i
        H_p_tmp1, time_tmp_1, time_tmp_2 = Cannon(H_p_tmp1, H_p_tmp2, index_x, index_y, n_x, n_y, n_local_x, n_local_y)
        H_m_tmp1, time_tmp_1, time_tmp_2 = Cannon(H_m_tmp1, H_m_tmp2, index_x, index_y, n_x, n_y, n_local_x, n_local_y)
        exp_H_p += H_p_tmp1 / factorial
        exp_H_m += H_m_tmp1 / factorial
        H_p_tmp2 = copy.deepcopy(H_p_backup)
        H_m_tmp2 = copy.deepcopy(H_m_backup)

    exp_H_p_backup = copy.deepcopy(exp_H_p)
    exp_H_m_backup = copy.deepcopy(exp_H_m)

    end = time.time()
    time2 += end - start
    comm.barrier()

    comm.barrier()
    time_QUE = []
    time_QUE_1 = []
    time_QUE_2 = []
    time_QME = []
    time_QME_1 = []
    time_QME_2 = []


    ##################################################
    ##########      Unitary evolution       ##########
    ##################################################

    for i in range(iteration):

        comm.barrier()
        start = time.time()

        rho_inter, time_QUE_tmp_1, time_QUE_tmp_2 = Cannon(exp_H_m, rho, index_x, index_y, n_x, n_y, n_local_x, n_local_y)
        
        end = time.time()
        time_QUE.append(end - start)
        for j in range(len(time_QUE_tmp_1)):
            time_QUE_1.append(time_QUE_tmp_1[j])
        for j in range(len(time_QUE_tmp_2)):
            time_QUE_2.append(time_QUE_tmp_2[j])
        comm.barrier()

        start = time.time()
        
        rho, time_QUE_tmp_1, time_QUE_tmp_2 = Cannon(rho_inter, exp_H_p, index_x, index_y, n_x, n_y, n_local_x, n_local_y)

        end = time.time()
        time_QUE.append(end - start)
        for j in range(len(time_QUE_tmp_1)):
            time_QUE_1.append(time_QUE_tmp_1[j])
        for j in range(len(time_QUE_tmp_2)):
            time_QUE_2.append(time_QUE_tmp_2[j])
        comm.barrier()

        exp_H_p = copy.deepcopy(exp_H_p_backup)
        exp_H_m = copy.deepcopy(exp_H_m_backup)

        rho_tmp = copy.deepcopy(rho)
    
        comm.barrier()
        start = time.time()

        rho, time_QME_tmp_1, time_QME_tmp_2 = QME(rho, rho_tmp, local_disses_hor, local_disses_ver, local_influxes_hor, local_influxes_ver, local_replace)

        end = time.time()
        time_QME.append(end - start)
        for j in range(len(time_QME_tmp_1)):
            time_QME_1.append(time_QME_tmp_1[j])
        for j in range(len(time_QME_tmp_2)):
            time_QME_2.append(time_QME_tmp_2[j])
        comm.barrier()

        # Diagonal of Hamiltonian
        if(index_x == index_y):
            tmp = []
            for j in range(n_local_x):
                tmp.append(rho[j, j])
            rho_ev.append(np.array(tmp))

    comm.barrier()
    TIME = 0.
    for i in range(len(time_QUE)):
        TIME += time_QUE[i]
    time3 += TIME
    TIME = 0.
    for i in range(len(time_QUE_1)):
        TIME += time_QUE_1[i]
    time3_1 += TIME
    TIME = 0.
    for i in range(len(time_QUE_2)):
        TIME += time_QUE_2[i]
    time3_2 += TIME
    TIME = 0.
    for i in range(len(time_QME)):
        TIME += time_QME[i]
    time4 += TIME
    TIME = 0.
    for i in range(len(time_QME_1)):
        TIME += time_QME_1[i]
    time4_1 += TIME
    TIME = 0.
    for i in range(len(time_QME_2)):
        TIME += time_QME_2[i]
    time4_2 += TIME

    if(rank == 0):
        group_time1 = []
        group_time1.append(time1)
        group_time2 = []
        group_time2.append(time2)
        group_time3 = []
        group_time3.append(time3)
        group_time3_1 = []
        group_time3_1.append(time3_1)
        group_time3_2 = []
        group_time3_2.append(time3_2)
        group_time4 = []
        group_time4.append(time4)
        group_time4_1 = []
        group_time4_1.append(time4_1)
        group_time4_2 = []
        group_time4_2.append(time4_2)
    comm.barrier()
    for i in range(1, size):
        if(rank == i):
            comm.send(time1, dest = 0, tag = 0)
            comm.send(time2, dest = 0, tag = 1)
            comm.send(time3, dest = 0, tag = 2)
            comm.send(time3_1, dest = 0, tag = 3)
            comm.send(time3_2, dest = 0, tag = 4)
            comm.send(time4, dest = 0, tag = 5)
            comm.send(time4_1, dest = 0, tag = 6)
            comm.send(time4_2, dest = 0, tag = 7)
        if(rank == 0):
            time1 = comm.recv(source = i, tag = 0)
            time2 = comm.recv(source = i, tag = 1)
            time3 = comm.recv(source = i, tag = 2)
            time3_1 = comm.recv(source = i, tag = 3)
            time3_2 = comm.recv(source = i, tag = 4)
            time4 = comm.recv(source = i, tag = 5)
            time4_1 = comm.recv(source = i, tag = 6)
            time4_2 = comm.recv(source = i, tag = 7)
            group_time1.append(time1)
            group_time2.append(time2)
            group_time3.append(time3)
            group_time3_1.append(time3_1)
            group_time3_2.append(time3_2)
            group_time4.append(time4)
            group_time4_1.append(time4_1)
            group_time4_2.append(time4_2)
    comm.barrier()
    if(rank == 0):
        if(ii == 0):
            np.save('group_time1.npy', group_time1)
            np.save('group_time2.npy', group_time2)
            np.save('group_time3.npy', group_time3)
            np.save('group_time3_1.npy', group_time3_1)
            np.save('group_time3_2.npy', group_time3_2)
            np.save('group_time4.npy', group_time4)
            np.save('group_time4_1.npy', group_time4_1)
            np.save('group_time4_2.npy', group_time4_2)
        elif(ii == N - 1):
            group_time1_old = np.load("group_time1.npy")
            group_time2_old = np.load("group_time2.npy")
            group_time3_old = np.load("group_time3.npy")
            group_time3_1_old = np.load("group_time3_1.npy")
            group_time3_2_old = np.load("group_time3_2.npy")
            group_time4_old = np.load("group_time4.npy")
            group_time4_1_old = np.load("group_time4_1.npy")
            group_time4_2_old = np.load("group_time4_2.npy")
            group_time1 += group_time1_old
            group_time2 += group_time2_old
            group_time3 += group_time3_old
            group_time3_1 += group_time3_1_old
            group_time3_2 += group_time3_2_old
            group_time4 += group_time4_old
            group_time4_1 += group_time4_1_old
            group_time4_2 += group_time4_2_old
            group_time1 /= N 
            group_time2 /= N 
            group_time3 *= float(1000 / iteration) / N 
            group_time3_1 *= float(1000 / iteration) / N 
            group_time3_2 *= float(1000 / iteration) / N 
            group_time4 *= float(1000 / iteration) / N 
            group_time4_1 *= float(1000 / iteration) / N 
            group_time4_2 *= float(1000 / iteration) / N 
            print(group_time1)
            print(group_time2)
            print(group_time3)
            print(group_time3_1)
            print(group_time3_2)
            print(group_time4)
            print(group_time4_1)
            print(group_time4_2)
            np.save('group_time1.npy', group_time1)
            np.save('group_time2.npy', group_time2)
            np.save('group_time3.npy', group_time3)
            np.save('group_time3_1.npy', group_time3_1)
            np.save('group_time3_2.npy', group_time3_2)
            np.save('group_time4.npy', group_time4)
            np.save('group_time4_1.npy', group_time4_1)
            np.save('group_time4_2.npy', group_time4_2)
        else:
            group_time1_old = np.load("group_time1.npy")
            group_time2_old = np.load("group_time2.npy")
            group_time3_old = np.load("group_time3.npy")
            group_time3_1_old = np.load("group_time3_1.npy")
            group_time3_2_old = np.load("group_time3_2.npy")
            group_time4_old = np.load("group_time4.npy")
            group_time4_1_old = np.load("group_time4_1.npy")
            group_time4_2_old = np.load("group_time4_2.npy")
            group_time1 += group_time1_old
            group_time2 += group_time2_old
            group_time3 += group_time3_old
            group_time3_1 += group_time3_1_old
            group_time3_2 += group_time3_2_old
            group_time4 += group_time4_old
            group_time4_1 += group_time4_1_old
            group_time4_2 += group_time4_2_old
            np.save('group_time1.npy', group_time1)
            np.save('group_time2.npy', group_time2)
            np.save('group_time3.npy', group_time3)
            np.save('group_time3_1.npy', group_time3_1)
            np.save('group_time3_2.npy', group_time3_2)
            np.save('group_time4.npy', group_time4)
            np.save('group_time4_1.npy', group_time4_1)
            np.save('group_time4_2.npy', group_time4_2)
    comm.barrier()
if(rank == 0):
    total_time = time.time() - total_start
    print("time: %d h %d m %.6f s" % ((total_time // 3600), ((total_time % 3600) // 60), ((total_time % 3600) % 60)))
