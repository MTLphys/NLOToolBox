# -*- coding: utf-8 -*-
"""
Definition of Propagator and main functionality of the program
@author: mattl
"""
import numpy as np 
import numba as nb 

II = np.eye(2)  
def main():
    print('Hello World')
  
#defining LUdecomposition
nb.jit()
def LUdecomposition():
    return 0 
#defining the first order unitary Crank propogator for the plasmonn
nb.jit()
def inv(A):
    a=0.0
    y=0.0
    i = 0 
    j = 0 

#defining the first order unitary Crank propogator for the plasmonn
nb.jit()
def firstorderaprx(E,E1i,E2i,dt): 
    H = np.array([[E1i,-E],[-E,E2i]],dtype=np.complex_)
    M = II-H*0.5j*dt
    N = II+H*0.5j*dt
    NI= np.linalg.inv(N)
    return np.matmul(NI,M)

def propagate(U,HR,dt,rho,GS):
    UD = np.conjugate(U)
    R = HR*(rho-GS)
    #^intentionally elementwise multiplication not matrix product as HR1 is applied 
    #to each component individually
    M = np.matmul(rho,UD)
    return np.matmul(U,M)-dt*R

if __name__ == "__main__" :
    main()
