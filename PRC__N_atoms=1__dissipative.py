##################################################
#             Miao Hui-hui, aspirant             #
# Faculty CMC, Lomonosov Moscow State University #
#                hhmiao@cs.msu.ru                #
##################################################


# -*- coding:utf-8 -*-
import numpy as np
import scipy.linalg as scpl
import time
import copy
np.seterr(divide = 'ignore', invalid = 'ignore')

N = 1

total_start = time.time()

time1 = 0.
time2 = 0.
time3 = 0.
time4 = 0.


##################################################
##########         Dissipation          ##########
##################################################

# Lindblad equation for dissipation-influx process
def PRC(rho, rho1, gamma, a, b, l, s):
    rho[b][b] += gamma * rho1[a][a] * s
    for i in range(l):
        rho[a][i] -= 0.5 * gamma * rho1[a][i] * s
        rho[i][a] -= 0.5 * gamma * rho1[i][a] * s
    return rho

def PRC_Influx(rho, rho1, gamma, a, b, l, s, mu):
    rho[b][b] += gamma * rho1[a][a] * s
    rho[a][a] += mu * gamma * rho1[b][b] * s
    for i in range(l):
        rho[a][i] -= 0.5 * gamma * rho1[a][i] * s
        rho[b][i] -= 0.5 * mu * gamma * rho1[b][i] * s
    for i in range(l):
        rho[i][a] -= 0.5 * gamma * rho1[i][a] * s
        rho[i][b] -= 0.5 * mu * gamma * rho1[i][b] * s
    return rho

for ii in range(N):

    ##################################################
    ##########     GENERATION OF STATES     ##########
    ##################################################

    start = time.time()

    n_atom = 1 # Number of atoms
    list_states_tmp = []
    cnt1 = -1
    for i1 in range(2):
        cnt1 += 1
        cnt2 = -1
        for i2 in range(2):
            cnt2 += 1
            if(cnt1 + cnt2 <= 1):
                list_states_tmp.append([cnt1, cnt2])
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
    #print(list_states)
    print(len(list_states))
    #print(list_inters)
    #print(len(list_inters))
    #print(list_disses)
    print(len(list_disses))
    #exit()


    ##################################################
    ##########          PARAMETERS          ##########
    ##################################################

    hbar = 1. # Reduced Planck constant
    w = 10 ** 8 # Photon frequency
    g = 10 ** 6 # Strength
    gamma = 10 ** 6 # Dissipation intensity
    mu = 0.
    t = 2. * 10 ** -5 # Period
    iteration = 20000 # Interations
    d_tau = t / iteration # Time span
    exponent = 20 # Taylor series


    ##################################################
    ##########         HAMILITONIAN         ##########
    ##################################################

    # Build the Hamiltonian with the help of above groups

    H_m = np.zeros((len(list_states), len(list_states)))
    for i in range(len(list_inters)):
        H_m[list_inters[i][0]][list_inters[i][1]] = g
    for i in range(len(list_states)):
        cnt = 0
        for j in range(len(list_states[0])):
            cnt += list_states[i][j]
        H_m[i][i] = cnt * hbar * w
    #print(H_m)
    #exit()

    end = time.time()
    time1 += end - start

    # Build the initial density matrix
    rho_0 = np.zeros((len(list_states), len(list_states)))
    rho_0[0][0] = 1.
    rho = copy.deepcopy(rho_0)

    rho_ev = []
    x_mesh = []

    # Diagonal of Hamiltonian
    tmp = []
    for i in range(len(list_states)):
	    tmp.append(np.abs(rho[i, i].real))
    rho_ev.append(np.array(tmp))
    x_mesh.append(0)


    ##################################################
    ##########        Taylor series         ##########
    ##################################################

    start = time.time()

    H_p_backup =  1j * H_m * d_tau
    H_m_backup =  -1j * H_m * d_tau
    exp_H_p = np.eye(len(list_states), dtype = 'complex_')
    exp_H_m = np.eye(len(list_states), dtype = 'complex_')
    exp_H_p += 1j * H_m * d_tau
    exp_H_m += -1j * H_m * d_tau
    factorial = 1
    H_p_tmp1 = 1j * H_m * d_tau
    H_p_tmp2 = 1j * H_m * d_tau
    H_m_tmp1 = -1j * H_m * d_tau
    H_m_tmp2 = -1j * H_m * d_tau
    for i in range(2, exponent + 1):
        factorial *= i
        H_p_tmp1 = np.dot(H_p_tmp1, H_p_tmp2)
        H_m_tmp1 = np.dot(H_m_tmp1, H_m_tmp2)
        exp_H_p += H_p_tmp1 / factorial
        exp_H_m += H_m_tmp1 / factorial
        H_p_tmp2 = 1j * H_m * d_tau
        H_m_tmp2 = -1j * H_m * d_tau

    end = time.time()
    time2 += end - start

    time_QUE = []
    time_QME = []


    ##################################################
    ##########      Unitary evolution       ##########
    ##################################################

    cnt = 0

    for i in range(iteration):

        # Running progress
        #if((i + 1) % (iteration / 100) == 0):
        #    print("Running progress: %d%%" % cnt)
        #    cnt += 1

        start = time.time()

        rho = np.dot(exp_H_m, np.dot(rho, exp_H_p))

        end = time.time()
        time_QUE.append(end - start)

        rho_tmp = copy.deepcopy(rho)
    
        start = time.time()

        for j in range(len(list_disses)):
            rho = PRC_Influx(rho, rho_tmp, gamma, list_disses[j][0], list_disses[j][1], len(list_states), d_tau, mu)

        end = time.time()
        time_QME.append(end - start)

        # Diagonal of Hamiltonian
        if((i + 1) % 1 == 0):
            tmp = []
            for j in range(len(list_states)):
                tmp.append(rho[j, j].real)
            rho_ev.append(np.array(tmp))
            x_mesh.append((i + 1) * d_tau)

    TIME = 0.
    for i in range(len(time_QUE)):
        TIME += time_QUE[i]
    time3 += TIME
    TIME = 0.
    for i in range(len(time_QME)):
        TIME += time_QME[i]
    time4 += TIME

# Save
np.save('x_mesh.npy', x_mesh)
np.save('rho_ev.npy', rho_ev)
np.save('list_states.npy', list_states)

print("time: %d h %d m %.6f s" % (((time1 / N) // 3600), (((time1 / N) % 3600) // 60), (((time1 / N) % 3600) % 60)))
print("time: %d h %d m %.6f s" % (((time2 / N) // 3600), (((time2 / N) % 3600) // 60), (((time2 / N) % 3600) % 60)))
print("time: %d h %d m %.6f s" % (((time3 / N) // 3600), (((time3 / N) % 3600) // 60), (((time3 / N) % 3600) % 60)))
print("time: %d h %d m %.6f s" % (((time4 / N) // 3600), (((time4 / N) % 3600) // 60), (((time4 / N) % 3600) % 60)))

total_time = time.time() - total_start
print("time: %d h %d m %.6f s" % ((total_time // 3600), ((total_time % 3600) // 60), ((total_time % 3600) % 60)))
