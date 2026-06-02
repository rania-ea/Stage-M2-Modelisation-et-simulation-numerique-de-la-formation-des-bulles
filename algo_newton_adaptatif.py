# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:43:00 2026

@author: rania
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:27:22 2026

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
eta = 0.1 # distance minimale entre les deux phases
# enfaite plus le etat et petit, comme cette condition est lourd, minimize va la sur contidérais et afficher tout les minimaux entre les phases. 
## ensenble des fonction dont on aura besoin:
c= 25.75831
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
def D_fermer(x):
    cAp, cBp, cAm, cBm = x
    d = D(x)
    return np.array([d[0],d[1],d[2],cAm + cBm - c])

# # on ajoute les contrainte 
# constraints = [
#     {"type": "ineq", "fun": lambda x: x[0] - eps},
#     {"type": "ineq", "fun": lambda x: x[1] - eps},
#     {"type": "ineq", "fun": lambda x: x[2] - eps},
#     {"type": "ineq", "fun": lambda x: x[3] - eps},

#     {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[0] - b_B*x[1]},
#     {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[2] - b_B*x[3]},

#     # contrainte pour que (cap,cbp) différent de (cam,cbm):
#     # (x0,x1) doit être différent de (x2,x3)
#     {"type": "ineq", "fun": lambda x: (x[0]-x[2])**2 + (x[1]-x[3])**2 - eta**2},
# ]

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


def jac_D_fermer(x):
    cAp, cBp, cAm, cBm = x

    HAAp, HABp, HBBp = hess_f(cAp, cBp)
    HAAm, HABm, HBBm = hess_f(cAm, cBm)

    J = np.zeros((4, 4))

    
    J[0, :] = [HAAp, HABp, -HAAm, -HABm]

   
    J[1, :] = [HABp, HBBp, -HABm, -HBBm]

    J[2, 0] = -(cAp*HAAp + cBp*HABp)
    J[2, 1] = -(cAp*HABp + cBp*HBBp)
    J[2, 2] =  (cAm*HAAm + cBm*HABm)
    J[2, 3] =  (cAm*HABm + cBm*HBBm)
    J[3, :] = [0, 0, 1, 1]

    return J



eps2 = 1e-9
beta = 0.9
nmax = 1000

def newton(D, jac_num, init, e, nmax=100):
    x = np.array(init, dtype=float)

    for k in range(nmax):
        F = D(x)
        normF = np.linalg.norm(F)

        if normF < e:
            return x, normF, True

        J = jac_num(x)

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return x, normF, False

        alpha = 1.0
        x_new = x + alpha * delta

        while not domaine_ok(x_new):
            alpha *= beta
            x_new = x + alpha * delta

            if alpha < 1e-12:
                return x, normF, False

        x = x_new

    return x, np.linalg.norm(D(x)), False



#initialisation avec multi-start
# choisie un point au hasard et vérifie si bien dans la région des réalisable

def random_point():
    while True:
        x = np.random.uniform(0.1, 30, size=4)

        dist2 = (x[0] - x[2])**2 + (x[1] - x[3])**2

        if domaine_ok(x) and dist2 > eta**2:  # on ajoute les contrainte du domaine ok et la bonne distance entre les points ie on peut que les point soit bien ecarter
            return x

sols = []
borne = 1e-12
nmax = 10000

for j in range(100):
    print(j)
    # meilleur_res = None
    x0 = random_point()
    # print(i)
    if not domaine_ok(x0):  
        continue
    # on minimise la fonction D avec minimize
    res, val, ok = newton(D_fermer, jac_D_fermer, x0, borne, nmax)

    if ok:
        sols.append([res, val])
        
# res = meilleur_res        
# print("min avec jacobienne x =", res.x)
# print("D(x) =", D(res.x))
# print("1/2 ||D(x)||² =", fonction_moindre_carre(res.x))

print(len(sols))

sols_x = np.array([res for res, val in sols])
sols_fun = np.array([val for res, val in sols])

bords_ca = [0, 1/b_A, 0, 0]
bords_cb = [1/b_B, 0, 0, 1/b_B]


scatter = plt.scatter(sols_x[:,0], sols_x[:,1], c=sols_fun, cmap="viridis")
plt.plot(bords_ca, bords_cb, color='k', label='Frontière du domaine')

# plt.plot(sols[:, 2], sols[:, 3], 'x', label = 'cam, cbm')
plt.xlabel('cap')
plt.ylabel('cbp')
plt.colorbar(scatter, label="valeur")
plt.legend()
plt.title(f"Solutions trouvées pour eta = {eta}")
plt.show()
        
 ## Resultat         ( pour 25 init)

# min 1
##############[1.3, 16.2, 25.4, 0.6]


#[ 1.62827797 16.66856355 25.42372508  0.66068386]
# [ 1.88671102 17.1177461  25.58146147  0.73486463]
# [ 1.24719593 16.14854191 25.36188348  0.56536438] 
# [ 1.30169835 16.22566359 25.36609724  0.57847075]
# min 2
###############[7.3, 14.0, 4.86, 15.77]


#[ 7.30980884 14.0351591   4.85995539 15.76669544]
# [ 7.30821225 14.04442164  4.85844356 15.77607789]
# [ 7.3119359  14.02216515  4.86195542 15.75352174]
# [ 7.30863139 14.04203966  4.85884137 15.77366574]
#[ 7.30811234 14.04498396  4.85834864 15.77664726 ]
# [ 6.86064447 14.15804375  5.22522572 15.30930633]
# [ 6.8582206  14.12498994  5.2222401  15.2754541 ]
#[ 6.86001171 14.14696483  5.22441249 15.29797099]
# min 3
################[21.2, 4.15, 23.6, 2.42]


#[21.2125655   4.1515243  23.66504547  2.42371007]
# [21.16685644  4.14932335 23.62002305  2.42248417]
# [21.20530943  4.15124931 23.65789204  2.42358079]
# [20.64676278  3.97633768 23.12007331  2.27847578]

# min 3 mirroire
# [23.64413405  2.42327535 21.19134753  4.15065436]
# [23.63202572  2.42290462 21.17905169  4.15001735]
#[22.92654384  2.5883609  21.28393152  3.72933627]
# [23.19188287  2.6499571  21.55523076  3.79946563]
# [22.95796469  2.60028796 21.31634763  3.74269482]
# [22.98417933  2.60901854 21.34331559  3.75250715]




       
