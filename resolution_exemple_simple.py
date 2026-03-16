# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:51:31 2026

@author: rania
"""

import numpy as np
from scipy.optimize import fsolve

#paramétre 
a_AA = 2.0
a_AB = 1.0
a_BB = 1.5
b_A = 0.1
b_B = 0.2
k = 1.0
t = 300
p = 64

c_A_plus = 0.75
c_B_plus = 0
c_A_moins = -0.9
c_B_moins = 0

x = np.array([c_A_plus, c_B_plus, c_A_moins, c_B_moins])




def f1(x):
    return 4* x[0]*(x[0]**2 - 1 ) - 4* x[2]*(x[2]**2 - 1 ) 
def f2(x):
    return 2* x[1] -2*x[3]
def f3(x):
    return (x[0]**2 -1)**2 + x[1]**2 - 4* x[0]*(x[0]**2 - 1 )*x[0] -2*x[1]*x[1]- ((x[2]**2 -1)**2 + x[3]**2 - 4* x[2]*(x[2]**2 - 1 )*x[2] -2*x[3]*x[3])
def f4(x):                                                                                                                                                           
    return x[1] - p


def D(x):
    return np.array([f1(x), f2(x), f3(x), f4(x)])

def jac_num(x, h=1e-8):
    x = np.array(x, dtype=float)
    J = np.zeros((4, 4))
    Fx = D(x)
    

    for j in range(4):
        xh = x.copy()
        xh[j] += h
        J[:, j] = (D(xh) - Fx) / h

    return J  
init = x
def jacobienne(x):
    return np.array([
        [12*x[0]**2 - 4,  0,   -12*x[2]**2 + 4,  0],
        [0, 2, 0, -2],
        [-12*x[0]**3 + 4*x[0],  -2*x[1],  12*x[2]**3 - 4*x[2],  2*x[3]],
        [0, 1, 0,0] ])

def newton(D, jacobienne, init, e, nmax=500):
    x_old = np.array(init, dtype=float)
    k = 0

    while k < nmax:
        F = D(x_old)
        J = jacobienne(x_old)

        delta = np.linalg.solve(J, -F)
        x_new = x_old + delta

        if np.linalg.norm(delta) < e:
            return x_new

        x_old = x_new
        k += 1

    return x_old
print(newton(D,jac_num,init,10**(-2)),"newton avec jac num")
print(newton(D,jacobienne,init,10**(-2)),"newton avec jac a la main")
print( fsolve(D, x,fprime=jacobienne),"fsolve avec jac a la main")
print( fsolve(D, x,fprime=jac_num),"fsolve avec jac num")






