#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:52:03 2024

@author: ashilcusky
"""

import time
import math
import numpy as np
import numba
from matplotlib import pyplot as pl


''' This file calculates the energy EXACTLY, using 
<E> = sum_x |psi_x|^2 eL(x) 
        / (sum_x |psi_x|^2)
        
'''
t0 = time.time()



def IBITS(n,i):
    return ((n >> i) & 1)

def _int_to_state2(integer, N):
    state = np.ones(N)
    for bit in range(N):
        if IBITS(integer,bit) == 0:
            state[bit] = -1
    return state

def _state_to_int(state, N):
    x = (state+np.ones(N)*0.5)
    x = x.astype('int')
    return int(''.join(map(str,x[::-1])),base=2 )

    
def get_ALL_states(N):
    int_basis = []
    hbasis1D = []
    def bit_count(self):
        return bin(self).count("1")

    for i in range(2**N-1):
        # print(bit_count(i))
        # print(i)
        if bit_count(i) == int(N/2):
            # hbasis2D.append(i)
            int_basis.append(i)
            state = _int_to_state2(i,N)
            state = state*0.5
            hbasis1D.append(state)
    return hbasis1D, int_basis



def get_zIJ(Q, K, xlist):
    Nc = len(xlist)
    z = np.zeros((Nc,Nc))
    for i in range(Nc):
        for j in range(Nc):
            xI = xlist[i]
            xJ = xlist[j]
            qI = np.matmul(Q,xI)
            kJ = np.matmul(K,xJ)
            z[i,j] = np.dot(qI, kJ)/np.sqrt(L)
    return z

def get_alist(z):
    Nc = len(z[0,:])
    a = np.zeros(Nc)
    for i in range(Nc):
        aI = np.exp(-z[i,i])
        denom = 0
        for j in range(Nc):
            denom += np.exp(-z[i,j])
            # print(denom)
        aI /= denom
        a[i] = aI
    return a


def get_vtilde(V, xlist, alist):
    Nc = len(xlist)
    aVx = []
    ax = []
    Vx = []
    for i in range(Nc):
        aI = alist[i]
        xI = xlist[i] 
        ax.append(aI*xI)
        aVx.append(aI*np.matmul(V,xI))
        Vx.append(np.matmul(V,xI))
    vtilde = np.concatenate(aVx) # vtilde = aVx
    ax = np.concatenate(ax)
    Vx = np.concatenate(Vx)
    return vtilde, ax, Vx


def get_coeff(vtilde, W):
    return np.exp(vtilde.T @ W @ vtilde)


def get_eL(state, coeff, Q, K, V,W):
    N = len(state)
    L = len(Q[0,:])
    res = 0
    ssum = 0
    for i in range(N):
        res += state[i] * state[(i+1)%N]
        if (state[i] * state[(i+1)%N] < 0):
            state_new = state.copy()
            state_new[i] *= -1
            state_new[(i+1)%N] *= -1
            
            # xlist_new = list(get_clusters(state_new,L))
            xlist_new = np.reshape(state_new, (Nc,L))
            zIJ_new = get_zIJ(Q, K, xlist_new)
            alist_new = get_alist(zIJ_new)
            vtilde_new , ax, Vx = get_vtilde(V,xlist_new, alist_new)
            ssum += get_coeff(vtilde_new, W)/coeff
    return res - 0.5*ssum

def get_logder_W(vtilde):
    mat = np.outer(vtilde,vtilde)
    return mat

def get_logder_V(alist, xlist, V, W):
    L = len(V[0,:])
    Nc = len(alist)
    OV = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            for I in range(Nc):
                for J in range(Nc):
                    aI = alist[I]
                    aJ = alist[J]
                    xI = xlist[I]
                    xJ = xlist[J]
                    vI = V@xI
                    vJ = V@xJ
                    for k in range(L):
                        OV[i,j] += aI*xI[j]*W[int(I*L+i), int(J*L+k)]*aJ*vJ[k] \
                            + aI*vI[k]*W[int(I*L+k),int(J*L+i)]*aJ*xJ[j]
    return OV

def get_logder_QK(alist,xlist,Q,K,V,W, z):
    L = len(Q[0,:])
    Nc = len(alist)
    OQ = np.zeros((L,L))
    OK = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            daI_dQ_lst = np.zeros(Nc)
            daI_dK_lst = np.zeros(Nc)
            for I in range(Nc):
                aI = alist[I]
                xI = xlist[I]
                qI = Q@xI
                kI = K@xI
                daI_dQ_lst[I] = -aI * xI[j]*kI[i]/np.sqrt(L)
                daI_dK_lst[I] = -aI * xI[j]*qI[i]/np.sqrt(L)
                for J in range(Nc):
                    kJ = K@xlist[J]
                    qJ = Q@xlist[J]
                    daI_dQ_lst[I] += (aI)**2 * np.exp(-z[I,J] + z[I,I]) * xI[j]*kJ[i] / np.sqrt(L)
                    daI_dK_lst[I] += (aI)**2 * np.exp(-z[I,J] + z[I,I]) * xI[j]*qJ[i] / np.sqrt(L)
                    
            for I in range(Nc):
                for J in range(Nc):
                    for k in range(L):
                        for l in range(L):
                            vI = V@xlist[I]
                            vJ = V@xlist[J]
                            aI = alist[I]
                            aJ = alist[J]
                            daIdQ = daI_dQ_lst[I]
                            daJdQ = daI_dQ_lst[J]
                            daIdK = daI_dK_lst[I]
                            daJdK = daI_dK_lst[J]
                            OQ[i,j] += W[int(I*L)+k,int(J*L+l)]*(daIdQ*vI[k] * aJ*vJ[l] + \
                                                                  aI*vI[k] * daJdQ*vJ[l])
                            OK[i,j] += W[int(I*L)+k,int(J*L+l)]*(daIdK*vI[k] * aJ*vJ[l] + \
                                                                  aI*vI[k] * daJdK*vJ[l])
    # print()
    return OQ, OK

def sampler(states_list, Q, K, V, W):
    N = len(W[0,:])
    L = len(Q[0,:])
    Nc = N // L
    
    E = 0
    coeffs_sum = 0
    OW = np.zeros((N,N), dtype = np.cdouble)
    H_OW = np.zeros((N,N), dtype = np.cdouble)
    
    OV = np.zeros((L,L), dtype = np.cdouble)
    H_OV = np.zeros((L,L), dtype = np.cdouble)
    
    Oq = np.zeros((L,L), dtype = np.cdouble)
    HOq = np.zeros((L,L), dtype = np.cdouble)
    Ok = np.zeros((L,L), dtype = np.cdouble)
    HOk = np.zeros((L,L), dtype = np.cdouble)
    for state in states_list:
        # xlist = list(get_clusters(state,L))
        xlist = np.reshape(state, (Nc, L))
        z = get_zIJ(Q,K,xlist)
        alist = get_alist(z)
        vtilde,ax,Vx = get_vtilde(V,xlist,alist)

        c = get_coeff(vtilde, W)
        # print('c = ', c)
        tmp_energy = get_eL(state, c, Q, K, V,W)
        
        
        coeffs_sum += c**2
        E += tmp_energy * c**2
        
        # gradient stuff for W
        tmp_logderW = get_logder_W(vtilde)
        OW += tmp_logderW * c**2
        H_OW += np.conjugate(tmp_logderW) * tmp_energy * c**2
        
        # gradient stuff for V
        tmp_logderV = get_logder_V(alist, xlist, V, W)
        OV += tmp_logderV * c**2
        H_OV += np.conjugate(tmp_logderV) * tmp_energy * c**2
        
        # gradient stuff for Q and K

        tmp_logderQ, tmp_logderK = get_logder_QK(alist,xlist,Q,K,V,W, z)
        Oq += tmp_logderQ * c**2
        HOq += np.conjugate(tmp_logderQ) * tmp_energy * c**2
        Ok += tmp_logderK * c**2
        HOk += np.conjugate(tmp_logderK) * tmp_energy * c**2

    E /= coeffs_sum
    H_OW /= coeffs_sum
    OW /= coeffs_sum
    
    H_OV /= coeffs_sum
    OV /= coeffs_sum
    
    gradW = 2*H_OW - 2*E*OW
    gradV = 2*H_OV - 2*E*OV 
    
    
    ### 
    
    HOq /= coeffs_sum
    Oq /= coeffs_sum
    
    HOk /= coeffs_sum
    Ok /= coeffs_sum
    
    gradQ = 2*HOq - 2*E*Oq
    gradK = 2*HOk - 2*E*Ok
    return E, gradW, gradV, gradQ, gradK




N = 6

L = 2
Nc = N // L

if (Nc * L != N):
    print('\n!!!ERROR!!!\n WRONG SYSTEM OR CLUSTER SIZE/NUMBER')



Q = np.random.rand(L,L)
K = np.random.rand(L,L) 

W = np.random.rand(N,N)
V = np.random.rand(L,L)

# Q = np.eye(L)
# K = np.eye(L)
# W = np.eye(N)
# V = np.eye(L)

states_list, int_list = get_ALL_states(N)

# t = time.time()
# print('Time to generate states list: ', t-t0)


# E, gradW, gradV, gradQ, gradK  = sampler(states_list, Q, K, V, W)
# print('E = ', E)
# print(gradW)
# E0 = -3.6510934089371783 # right answer for N=8
E0 = -2.8027756106027173 # right answer for N=6
if 0:
    yy = []
    
    lam = 0.2
    # lam1 = 0.02
    
    for i in range(200):
        E, gradW, gradV, gradQ, gradK  = sampler(states_list, Q, K, V, W)
        # E, gradW, gradV = sampler2(states_list, Q, K, V, W)
        print(i)
        print('E = ', E)
        W = W - lam * gradW
        V = V - lam * gradV
        Q = Q - lam * gradQ
        K = K - lam * gradK

        
        yy.append(E)
        
    pl.figure()
    pl.plot(yy)
    pl.title('Gradient Descent \n lam=%.2f\nN=%i'%(lam,N))
    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.hlines(E0, 0, len(yy), color='r', linestyles='--')
    # pl.ylim(-2.85,-2.75)
    pl.show()
    
    print('\nE/N = ', E/N)
    

# pl.hlines(E0, 0, len(yy), color='r', linestyles='--')

# E, gradW, gradV,gradQ, gradK = sampler(states_list, Q, K, V, W)


# for state in states_list:
#     xlist = np.reshape(state, (Nc, L))
#     z = get_zIJ(Q, K, xlist)
#     alist = get_alist(z)
#     vtilde, ax, Vx = get_vtilde(V,xlist, alist)
#     c = get_coeff(vtilde, W)
#     print(c)
# print('E = ',E)
if 1: 
    yy = []
    
    lam = 0.2
    # lam1 = 0.02
    gam = 0.9
    
    pV = 0
    pW = 0
    pQ = 0
    pK = 0
    for i in range(300):
        # E, gradW, gradV, gradQ, gradK = sampler2(states_list, Q, K, V, W)
        E, gradW, gradV,gradQ, gradK = sampler(states_list, Q, K, V, W)
        print(i)
        print('E = ', E)
        pW = gam*pW + lam*gradW
        pV = gam*pV + lam*gradV
        
        W = W - pW
        V = V - pV
        
        pQ = gam*pQ + lam*gradQ
        pK = gam*pK + lam*gradK
        Q = Q - pQ
        K = K - pK
        
        yy.append(E)
        
        
    pl.figure()
    pl.plot(yy)
    # pl.ylim(.0002, .0004)
    pl.title('Gradient Descent w/ Momentum \n N=%i'%N)
    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.hlines(E0, 0, len(yy), color='r', linestyles='--')
    pl.show()
    
    
# yy = np.asarray(yy) + E0*np.ones(len(yy))
# pl.figure()
# pl.plot(yy)
# # pl.ylim(.0002, .0004)
# pl.ylabel('Energy')
# pl.xlabel('Iteration')
# pl.show()
    
t = time.time()
print('\n N= ', N)
print('Runtime: ', t-t0)


if 0:
    x = [6,8,10,12,14]
    y = [.02, .11, .63, 3.48, 17.83]
    
    xx = np.linspace(min(x), max(x), 100)
    
    def get_f(x):
        return np.exp(x)
    pl.figure()
    pl.plot(x,y)
    pl.ylabel('Runtime (s)')
    pl.xlabel('N = # of sites')
    # pl.plot(xx, get_f(xx))
    pl.show()
