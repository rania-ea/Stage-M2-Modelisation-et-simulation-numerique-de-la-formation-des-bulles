# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:07:26 2026

@author: rania
"""

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

a_AA = 6.41
a_AB = 4.0
a_BB = 0.92
b_A = 0.036
b_B = 0.051
k_2 = 12.2
t = 1.0

eps = 1e-10

alpha = 0.1 # le paramétre de relaxation ###ATTENTION il faut que le alpha soit petit 

## ensenble des fonction dont on aura besoin:
    
def domaine_ok(x):
    cAp, cBp, cAm, cBm = x
    dp = 1 - b_A*cAp - b_B*cBp
    dm = 1 - b_A*cAm - b_B*cBm
    return min(cAp, cBp, cAm, cBm, dp, dm) > eps    #retourne un bool

def energie(c_A, c_B):
    d = 1 - b_A*c_A - b_B*c_B
    return (
        -a_AA*c_A**2
        -2*a_AB*c_A*c_B
        -a_BB*c_B**2
        + k_2*t*(c_A*np.log(c_A/d) + c_B*np.log(c_B/d))
    )


def mu_A(c_A, c_B):
    d = 1 - b_A*c_A - b_B*c_B
    return (
        -2*a_AA*c_A
        -2*a_AB*c_B
        + k_2*t*(np.log(c_A/d) + 1 + b_A*(c_A+c_B)/d)
    )


def mu_B(c_A, c_B):
    d = 1 - b_A*c_A - b_B*c_B
    return (
        -2*a_AB*c_A
        -2*a_BB*c_B
        + k_2*t*(np.log(c_B/d) + 1 + b_B*(c_A+c_B)/d)
    )

def D(x):
    cAp, cBp, cAm, cBm = x

    return np.array([
        mu_A(cAp, cBp) - mu_A(cAm, cBm),
        mu_B(cAp, cBp) - mu_B(cAm, cBm),
        energie(cAp, cBp)
        - mu_A(cAp, cBp)*cAp
        - mu_B(cAp, cBp)*cBp
        - (
            energie(cAm, cBm)
            - mu_A(cAm, cBm)*cAm
            - mu_B(cAm, cBm)*cBm
        )
    ])
# on pose f la fonction "distance eta"

def f(x):
    return  (x[0]-x[2])**2 + (x[1]-x[3])**2



def fonction_moindre_carre_relaxe(x):
    if not domaine_ok(x):
        # print("domaine non ok")
        return 1e100   # une grand valeur quelqu'onque 
    return 0.5*np.dot(D(x),D(x)) + ( alpha/2 ) * 1/( f(x) )

# on ajoute les contrainte 
constraints = [
    {"type": "ineq", "fun": lambda x: x[0] - eps},
    {"type": "ineq", "fun": lambda x: x[1] - eps},
    {"type": "ineq", "fun": lambda x: x[2] - eps},
    {"type": "ineq", "fun": lambda x: x[3] - eps},

    {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[0] - b_B*x[1]},
    {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[2] - b_B*x[3]},

]

def hess_f(c_A, c_B):
    d = 1 - b_A*c_A - b_B*c_B
    s = c_A + c_B

    H_AA = -2*a_AA + k_2*t*(
        1/c_A + b_A/d + b_A*(d + b_A*s)/d**2
    )

    H_BB = -2*a_BB + k_2*t*(
        1/c_B + b_B/d + b_B*(d + b_B*s)/d**2
    )

    H_AB = -2*a_AB + k_2*t*(
        b_B/d + b_A*(d + b_B*s)/d**2
    )

    return H_AA, H_AB, H_BB


def jac_D(x):
    cAp, cBp, cAm, cBm = x

    HAAp, HABp, HBBp = hess_f(cAp, cBp)
    HAAm, HABm, HBBm = hess_f(cAm, cBm)

    J = np.zeros((3, 4))

    
    J[0, :] = [HAAp, HABp, -HAAm, -HABm]

   
    J[1, :] = [HABp, HBBp, -HABm, -HBBm]

    J[2, 0] = -(cAp*HAAp + cBp*HABp)
    J[2, 1] = -(cAp*HABp + cBp*HBBp)
    J[2, 2] =  (cAm*HAAm + cBm*HABm)
    J[2, 3] =  (cAm*HABm + cBm*HBBm)

    return J
# il faut modifier la jabienne pour quel corésponde a la version relaxe
### on definie les vecteur des derivé partile du terme de relaxation

def g(a,b,c):
    return ( 2 * (b - a) / c**2 ) 
    

def gradiant_term_relaxe(x):
    return alpha /2 * np.array([
        g(x[0],x[2],f(x)),
        g(x[1],x[3],f(x)),
        g(x[2],x[0],f(x)),
        g(x[3],x[1],f(x))
    ])


def jac_fonction_moindre_carre_relaxe(x):
    if not domaine_ok(x):
        return np.ones(4)*1e30
    return jac_D(x).T @ D(x)  + gradiant_term_relaxe(x)



#initialisation avec multi-start
# choisie un point au hasard et vérifie si bien dans la région des réalisable

def random_point():
    while True:
        x = np.random.uniform(0.1, 30, size=4)


        if domaine_ok(x) :  # on ajoute les contrainte du domaine ok 
            return x

sols = []
borne = 1e-3
N = 1000
for j in range(N):
    print(j)
    # meilleur_res = None
    x0 = random_point()
    # print(i)
    if not domaine_ok(x0):  
        continue
    # on minimise la fonction D avec minimize
    res = minimize(
        fonction_moindre_carre_relaxe,
        x0,
        jac=jac_fonction_moindre_carre_relaxe,
        method="SLSQP",
        constraints=constraints,

    )
    if res.success and 0.5*np.dot(D(res.x), D(res.x)) < borne:   # ajoue du filtre due au terme de penalisation
        sols.append(res)
        
# res = meilleur_res        
# print("min avec jacobienne x =", res.x)
# print("D(x) =", D(res.x))
# print("1/2 ||D(x)||² =", fonction_moindre_carre(res.x))
    
sols = np.array(sols)


sols_x = np.array([sol.x for sol in sols])
sols_fun = np.array([sol.fun for sol in sols])

bords_ca = [0, 1/b_A, 0, 0]
bords_cb = [1/b_B, 0, 0, 1/b_B]


scatter = plt.scatter(sols_x[:,0], sols_x[:,1], c=sols_fun, cmap="viridis")
plt.plot(bords_ca, bords_cb, color='k', label='Frontière du domaine')

# plt.plot(sols[:, 2], sols[:, 3], 'x', label = 'cam, cbm')
plt.xlabel('cap')
plt.ylabel('cbp')
plt.colorbar(scatter, label="valeur")
plt.legend()
plt.title("Solutions trouvées pour minnimize relaxé")
plt.show()



       