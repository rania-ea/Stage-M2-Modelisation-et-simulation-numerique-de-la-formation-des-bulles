# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:46:20 2026

@author: rania
"""
import numpy as np
from scipy.optimize import minimize

a_AA = 6.41
a_AB = 4.0
a_BB = 0.92
b_A = 0.036
b_B = 0.051
k_2 = 12.2
t = 1.0

eps = 1e-10
eta = 3 # distance minimale entre les deux phases

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

def fonction_moindre_carre(x):
    if not domaine_ok(x):
        print("domaine non ok")
        return 1e100   # une grand valeur quelqu'onque 
    return 0.5*np.dot(D(x),D(x))

# on ajoute les contrainte 
constraints = [
    {"type": "ineq", "fun": lambda x: x[0] - eps},
    {"type": "ineq", "fun": lambda x: x[1] - eps},
    {"type": "ineq", "fun": lambda x: x[2] - eps},
    {"type": "ineq", "fun": lambda x: x[3] - eps},

    {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[0] - b_B*x[1]},
    {"type": "ineq", "fun": lambda x: 1 - eps - b_A*x[2] - b_B*x[3]},

    # contrainte pour que (cap,cbp) différent de (cam,cbm):
    # (x0,x1) doit être différent de (x2,x3)
    {"type": "ineq", "fun": lambda x: (x[0]-x[2])**2 + (x[1]-x[3])**2 - eta**2},
]





#initialisation avec multi-start
# choisie un point au hasard et vérifie si bien dans la région des réalisable

def random_point():
    while True:
        x = np.random.uniform(0.1, 20, size=4)
        dp = 1 - b_A*x[0] - b_B*x[1]
        dm = 1 - b_A*x[2] - b_B*x[3]
        if dp > 0 and dm > 0:
            return x

meilleur_res = None
for i in range(25):   #  ca verif 25 point qui marche
    x0 = random_point()
    print(i)
    if not domaine_ok(x0):  
        continue
    # on minimise la fonction D avec minimize
    res = minimize(
        fonction_moindre_carre,
        x0,
        method="SLSQP",
        constraints=constraints,
    )
    
    # on cherche le meilleur des resultat parmi tout ceux qu'on as trouvé
    if meilleur_res is None or res.fun < meilleur_res.fun:
            meilleur_res = res
res = meilleur_res        
print("x =", res.x)
print("D(x) =", D(res.x))
print("1/2 ||D(x)||² =", fonction_moindre_carre(res.x))
print("||D(x)|| =", np.linalg.norm(D(res.x)))      
        
        
# [ 3.67443227 14.75942722  6.23145568 13.19046314] avec 25 init        
        
        
        
        
        
        
        