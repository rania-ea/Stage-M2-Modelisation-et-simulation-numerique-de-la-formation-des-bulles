

import numpy as np
from matplotlib import pyplot as plt

a_AA = 6.41
a_AB = 4.0
a_BB = 0.92
b_A = 0.036
b_B = 0.051
k_2 = 12.2
t = 1.0

eps = 1e-10
dim = 1e-8 # diminution du triangle
alpha = 0 # le paramétre de relaxation ###ATTENTION il faut que le alpha soit petit 
# on ecrit les coeffifient a,b,c,d sur triangle
a_0 = 1/b_A
b_0 = 0
c_0= 0
d_0 = 1/b_B
# on reecrit les coeffifient a,b,c,d sur triangle diminuer : 
u = np.array([1,(a_0-c_0)/(d_0-b_0)])  
norm_u = np.linalg.norm(u)
lamda = (dim*(1+u[1]/norm_u)-b_0)/(b_0- d_0)

a = lamda * (a_0 -c_0) +a_0 - dim * u[0]/norm_u
b = dim
c= dim

lamda = (dim*(1+u[0]/norm_u)-a_0)/(a_0-c_0)
d = lamda * (b_0-d_0) +b_0 - dim * u[1]/norm_u

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

def fonction_moindre_carre(x):
    if not domaine_ok(x):
        # print("domaine non ok")
        return 1e100   # une grand valeur quelqu'onque 
    return 0.5*np.dot(D(x),D(x))

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

def jac_fonction_moindre_carre(x):
    if not domaine_ok(x):
        return np.ones(4)*1e30
    return jac_D(x).T @ D(x)

def proj_couple(x):
    
    if x[0] >= dim and x[1] >=dim and  x[1] <= (d-dim)/(dim-a)*(x[0]-dim) + d:
       return x 
    if x[0] <= dim and x[1] <= dim :
        return np.array([dim,dim])
    if x[0] <= a and dim <= x[0] and x[1] <= dim:
        return np.array([x[0],dim])
    if x[0] <= dim and d >= x[1] and x[1] >= dim:
        return np.array([dim,x[1]])
    if x[1] >=d and x[0]<= c + ((x[1] - d ) / u[1])*u[0]:
        return np.array([c,d])
    if x[0] >=a and x[1]<= b + ((x[0] - a ) / u[0])*u[1]:
        return np.array([a,b])  
    else:
        gamma = - ((c-a)*(a-x[0])+(d-b)*(b-x[1])) / ((c-a)**2+(d-b)**2)
        return gamma*np.array([c,d]) + (1-gamma)*np.array([a,b])
    
def proj(x):
    #if domaine_ok(x):
        #return x
    cp = np.array([x[0],x[1]])
    cm = np.array([x[2],x[3]])
    cp = proj_couple(cp)
    cm = proj_couple(cm)
    return np.array([cp[0],cp[1],cm[0],cm[1]])
    

eps2 = 1e-8

nmax = 1000
alpha0= 1e-8
def min_gradiant_projete(x):
    grad_J = jac_fonction_moindre_carre_relaxe

    
    n = 1
    c_n = x
    
    c_np1 = c_n - alpha0*grad_J(c_n)
    c_np1 = proj(c_np1)
    while np.linalg.norm(c_np1 - c_n) > eps2 and n < nmax :
        c_n = c_np1
        c_np1 = c_n - alpha0*grad_J(c_n)
        c_np1 = proj(c_np1)
        n += 1
    return c_np1
        
        

#initialisation avec multi-start
# choisie un point au hasard et vérifie si bien dans la région des réalisable

def random_point():
    while True:
        x = np.random.uniform(0.1, 30, size=4)


        if domaine_ok(x) :  # on ajoute les contrainte du domaine ok 
            return x

sols = []
borne = 1

relaxation = True

N = 1000
for j in range(N):
    print(j)
    # meilleur_res = None
    x0 = random_point()
    # print(i)
    if not domaine_ok(x0):  
        continue
    
    
    # on minimise la fonction D avec minimize
    res = min_gradiant_projete(x0)
    
    
    
    
    
    val = fonction_moindre_carre(res)
    print("val =", val, "res =", res, "D =", D(res))
    if val < borne:
        sols.append(res)
            
# res = meilleur_res        
# print("min avec jacobienne x =", res.x)
# print("D(x) =", D(res.x))
# print("1/2 ||D(x)||² =", fonction_moindre_carre(res.x))
    
sols = np.array(sols)


sols_x = sols
sols_fun = np.array([fonction_moindre_carre(sol) for sol in sols])  

bords_ca = [0, 1/b_A, 0, 0]
bords_cb = [1/b_B, 0, 0, 1/b_B]


scatter = plt.scatter(sols_x[:,0], sols_x[:,1], c=sols_fun, cmap="viridis")
plt.plot(bords_ca, bords_cb, color='k', label='Frontière du domaine')

# plt.plot(sols[:, 2], sols[:, 3], 'x', label = 'cam, cbm')
plt.xlabel('cap')
plt.ylabel('cbp')
plt.colorbar(scatter, label="valeur")
plt.legend()
if relaxation:
    plt.title("Solutions trouvées pour gradiant projeté relaxé")
else:
    plt.title("Solutions trouvées pour gradiant adaptatif")
plt.show()



       