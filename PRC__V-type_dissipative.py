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

    list_disses = []
    list_disses.append([7, 5])
    list_disses.append([7, 3])
    list_disses.append([6, 1])
    list_disses.append([5, 0])
    list_disses.append([4, 2])
    list_disses.append([3, 0])

    ##################################################
    ##########          PARAMETERS          ##########
    ##################################################

    hbar = 1. # Reduced Planck constant
    w_20 = 10 ** 8 # Photon frequency
    w_10 = 5 * 10 ** 7 # Photon frequency
    g_20 = 10 ** 6 # Strength
    g_10 = 5 * 10 ** 5 # Strength
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

    H_m = np.zeros((8, 8))
    H_m[5][2] = g_20
    H_m[2][5] = g_20
    H_m[7][4] = g_20
    H_m[4][7] = g_20
    H_m[3][1] = g_10
    H_m[1][3] = g_10
    H_m[7][6] = g_10
    H_m[6][7] = g_10
    H_m[0][0] = 0.
    H_m[1][1] = hbar * w_10
    H_m[2][2] = hbar * w_20
    H_m[3][3] = hbar * w_10
    H_m[4][4] = hbar * (w_20 + w_10)
    H_m[5][5] = hbar * w_20
    H_m[6][6] = hbar * (w_20 + w_10)
    H_m[7][7] = hbar * (w_20 + w_10)
    #print(H_m)
    #exit()

    # Build the initial density matrix
    rho_0 = np.zeros((8, 8))
    rho_0[7][7] = 1.
    rho = copy.deepcopy(rho_0)

    rho_ev = []
    x_mesh = []

    # Diagonal of Hamiltonian
    tmp = []
    for i in range(8):
	    tmp.append(np.abs(rho[i, i].real))
    rho_ev.append(np.array(tmp))
    x_mesh.append(0)


    ##################################################
    ##########        Taylor series         ##########
    ##################################################

    start = time.time()

    H_p_backup =  1j * H_m * d_tau
    H_m_backup =  -1j * H_m * d_tau
    exp_H_p = np.eye(8, dtype = 'complex_')
    exp_H_m = np.eye(8, dtype = 'complex_')
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
            rho = PRC_Influx(rho, rho_tmp, gamma, list_disses[j][0], list_disses[j][1], 8, d_tau, mu)

        end = time.time()
        time_QME.append(end - start)

        # Diagonal of Hamiltonian
        if((i + 1) % 1 == 0):
            tmp = []
            for j in range(8):
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

print("time: %d h %d m %.6f s" % (((time2 / N) // 3600), (((time2 / N) % 3600) // 60), (((time2 / N) % 3600) % 60)))
print("time: %d h %d m %.6f s" % (((time3 / N) // 3600), (((time3 / N) % 3600) // 60), (((time3 / N) % 3600) % 60)))
print("time: %d h %d m %.6f s" % (((time4 / N) // 3600), (((time4 / N) % 3600) // 60), (((time4 / N) % 3600) % 60)))

total_time = time.time() - total_start
print("time: %d h %d m %.6f s" % ((total_time // 3600), ((total_time % 3600) // 60), ((total_time % 3600) % 60)))
