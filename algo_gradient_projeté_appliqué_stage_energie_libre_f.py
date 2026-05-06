# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:32:15 2026

@author: rania
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:32:03 2026

@author: rania
"""
#algorithm du gradient projete pour dimention n fini
import numpy as np



#paramétre 
a_AA = 6.41
a_AB = 4
a_BB = 0.92
b_A = 0.036
b_B = 0.051
k_2 = 12.2
t = 1



# Ajout des  variables d'ecart pour saturais les contrainte

epsilon_1 = 1e-12
epsilon_2 = 1e-12
epsilon_3 = 1e-12


# INITIALISATION 
k = 0
epsilon = 1e-10
delta = 1e-13 # si t_l ,t_r sont trop proche
# on pose x**0
x_k = np.array([2.64154598e+01 ,2.49396886e-08])  # on a choisie un point de déprt qui soit dans la région des realisable

n= np.size(x_k)

def f(x): # fonction cout
    c_A, c_B = x
    denom = 1 - c_A*b_A - c_B*b_B
    return (
        - a_AA * c_A**2
        - 2 * a_AB * c_A * c_B
        - a_BB * c_B**2
        + k_2 * t * (c_A * np.log(c_A / denom) + c_B * np.log(c_B / denom)))




# definition des contrainte
A = np.array([[-1,0],[0,-1],[b_A,b_B]])
b = np.array([-epsilon_1 ,-epsilon_2 ,1- epsilon_3 ])

I_2 = [0,1,2]# on pose I_2 ie les indice des contrainte inactive


# calcule du gradiant de f 
# def grad_num(f, x, eps=10**(-6)):
#     x = np.asarray(x, dtype=float)
#     n = len(x)
#     g = np.zeros(n)
#     for i in range(n):
#         h = eps * max(abs(x[i]), 1.0)
#         xh1 = np.copy(x); xh1[i] += h
#         xh2 = np.copy(x); xh2[i] -= h
#         g[i] = (f(xh1) - f(xh2)) / (2*h)
#     return g

def potentiel_homogene_A(c_A, c_B):
    denom = 1 - b_A*c_A - b_B*c_B
    return (
        -2*a_AA*c_A
        -2*a_AB*c_B
        + k_2*t*(
            np.log(c_A / denom)
            + 1
            + b_A*(c_A + c_B)/denom))

def potentiel_homogene_B(c_A, c_B):
    denom = 1 - b_A*c_A - b_B*c_B
    return (
        -2*a_AB*c_A
        -2*a_BB*c_B
        + k_2*t*(
            np.log(c_B / denom)
            + 1
            + b_B*(c_A + c_B) / denom))
       

def grad_num_main(x_k):
    c_A, c_B = x_k
    grad_F = np.array([
        potentiel_homogene_A(c_A, c_B),
        potentiel_homogene_B(c_A, c_B)
    ])
    return grad_F 


for k in range(100):
    # 1 CALCULER I_0(x**k) ie ensemble des contraintes active en x**k
    
    b_1 = A @ x_k
    i = 0
    I_0 = []
    #print(I_0)
    for i in range(np.size(b)):
        if np.abs(b_1[i] - b[i]) <= 10**(-12):
            I_0 = np.append(I_0,i) 
    #print(I_0,'I_0')
         
    # 2 FORMER A_0 puis on vérifie si le rang de A_0 est respecter
       
    A_0 = np.empty(np.size(x_k)*np.size(I_0)).reshape(np.size(I_0),np.size(x_k))
    
    
    
    
    
    for i in range (len(I_0)):
        A_0[i,:] = A[int(I_0[i]),:]
        #print(A_0,'A_0')

    
    # 3 CALCULE P_0
    normp = 0
    test_break=False
    while normp < epsilon:
        if len(A_0) == 0 :
            P_0 = np.identity(n)
        else:
            P_0 = np.identity(n) - np.transpose(A_0) @ np.linalg.pinv( A_0 @ np.transpose(A_0) ) @ A_0
        # print(P_0,'P_k')
        
        
        
        # 4 CALCULE grad_f( x_k) , p**k et d**k
        
        h = 10**(-100) # pas (petit)
        
        

        gradf = grad_num_main(x_k)
        
        p_k = - P_0 @ gradf
        # print(p_k,'p_k')
        normp = np.linalg.norm(p_k) #norme euclidienne
        
        d_k = p_k / normp
        #print(d_k,'d_k')
        
        # 5 CALCULE DE u_k
        if len(A_0) == 0:
            u_k = np.array([])  # pas de contraintes actives → aucun multiplicateur
            u_k_i = 0
        else:
            print(A_0,'A_')
            u_k = np.linalg.pinv( A_0 @ np.transpose(A_0) ) @ A_0 @ gradf 
            #print(u_k,'u_k')
            u_k_i = np.min(u_k) #on pose u_k_i la plus petite des coordoner de u_k
        
        
        # 6 CONDITION 1 : D'ARRET
        print(normp,'normp',epsilon)
        print(u_k_i, 'u_k_i',-epsilon)
        if normp < epsilon and u_k_i >= -epsilon :
            
            print( 'STOP')
            test_break = True
            break
        
            
        # 7 CONDITION 2 :
        if normp < epsilon and u_k_i < -epsilon :
            i = np.argmin(u_k)
            A_0 = np.delete(A_0,i,0)
            #print('passer par 7')
            #print(A_0)
            
    if test_break == True :
            break    
     # 8 CALCULE T 

     # trouve T l'ensemble des t > 0 tel que x_k + t* d_k est dans X
     
    T=10**6 # on choisie arbritérment un t trés grand pour commencer a trouver le plus petit des t
    for i in I_2:
        term1 = b[i] - A[i,:] @ x_k
        term2 = A[i,:] @ d_k
        if term2 > 10**(-12) and T > term1 / term2:
            T = term1 / term2 
    T = max(0.0, 0.99 * T) # pour éviter de sortir du maxx
    #print(T,'T')
     
    # 9 CALCULE gradf2 et t_k AVEC ALGO 1
    i = 0
    t_i = min(T,1)
    #print(t_i)
    t_l = 0
    t_r = T
    beta1 = 10**(-4)
    beta2 = 0.99
    sg = f(x_k + t_i* d_k)
    sd = f(x_k) + t_i * beta1 * np.transpose(gradf) @ d_k
     
    gradf1 = grad_num_main( x_k + t_i * d_k)

    rg = np.transpose(gradf1) @ d_k
    rd = beta2 * np.transpose(gradf) @ d_k
    while (sg > sd or rg < rd) and i<100:
         if np.abs(t_l - t_r) < delta:
             break
         if sg > sd :
             #print(t_i,t_l,t_r,'prout1')
             t_r = t_i
             t_i = (t_l + t_r) / 2
         if sg <= sd and rg < rd :
             #print(sg,sd,rd,rg,' prout2')
             #print(t_l,t_r)
             t_l = t_i
             t_i = (t_l + t_r) / 2
         i= i+1      ### attention indentation
         sg = f(x_k + t_i* d_k)
         sd = f(x_k) + t_i * beta1 * np.transpose(gradf) @ d_k
         
         gradf1 = grad_num_main( x_k + t_i * d_k)
         rg = np.transpose(gradf1) @ d_k
         rd = beta2 * np.transpose(gradf) @ d_k
    # print(i)
    #10 INCREMENTATION DE x_k
    x_k = x_k + t_i * d_k
    x_k[0] = max(x_k[0], epsilon_1)  # pour eviter les osilations
    x_k[1] = max(x_k[1], epsilon_2)
    print(x_k,'x_k')
    print(f(x_k),"fonction evaluée au point final")
    #print(t_i)
    #print(d_k)
    #print(T)
    
print(f(np.array([26.485001,0.000000146])),"fonction evaluée au point trouver a la main ")
