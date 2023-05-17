# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:43:27 2022
Simulation of photon echo process generation code, Run first
@author: mattl
"""
#import packages
import numpy as np #import Math resources 
from scipy.io import savemat #import IO 
from NLOToolBox.Efield import Ef #import user defined Efield function
from NLOToolBox.Propagation import firstorderaprx, propagate #import user defined Propagator functions 

#Moving on to the coupled system simulation 
import time
#set up a timer 
start = time.time()#Timing Code execution 
Nest = 120
#assigning the properties of the plasmon 
w1 = np.pi*2.0*.36 #gap energy
T11 = 200.0        #damping constant 
T21 =  50.0        #Dephasing Constant
mu1 = 1.0          #Efield coupling constants 
G1 = .03
E11=0.0            #Ground Energy for plasmon .
E21=np.asarray([E11+w1+G1*np.random.normal()-G1/2 for i in range(Nest)])            #first excited energy For Plasmon
xi1 = 0.00         #System Coupling constant


#assigning the properies of the Exciton 
w2 = np.pi*2.0*.36#gap energy
T12 = 3000.0      #damping constant 
T22 = 1230.0      #Dephasing Constant
mu2 = 1.0         #Efield coupling constants
G2 = .03
E12=0.0           #Ground Energy for EXCITON
E22=np.asarray([E12+w2+G2*np.random.normal()- G2/2 for i in range(Nest)])            #first excited energy For EXCITON
xi2 = 0.00        #System Coupling constant

#generating time space (pulse train and decay time space)
t = np.linspace(0,4000,num=50000) #time 

dt = t[1]-t[0] #time differential 

#set up for the Efield
th1 = np.pi/2
th2 = np.pi 
E0 = .015
wl1 = np.pi*2.0*.36+(np.pi*2.0)*(255.0e-12)#first pulse Frequency 
wl2 = np.pi*2.0*.36+(np.pi*2.0)*(1225.0e-12)#second Pulse Frequency 
wl3 = 0#third Pulse Frequency
wlref = np.pi*2.0*.36
Pulse = True#pulse not cw 
Dt1=600 # second pulse @300 fs delay 
Dt2=0 #no first pulse 


wlexp = 2*wl2-wl1-wlref 
nperiod =np.round(np.pi*2/(wlexp*1.25e7)) #number of points per ~12.5 nsec period
print('number of steps per period:',nperiod)
tstep= np.pi*2/(nperiod*wlexp)# integer step for a ~12.5 nsecond period
print('length of step ',tstep*1e-6,'ns')
cperiod =nperiod*20*tstep #full collection period 
print('total collection time',cperiod*1e-6,'ns')
DT = np.arange(0,cperiod,tstep) #Pulse train
#Data Storage Variables
DR  = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize plasmon excited state  
R12 = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize plasmon polarization 
DS  = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton excited state
S12 = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton polarization

#ground state conditions
GS1= np.array([[1,0],[0,0]],dtype = np.complex128)#set plasmon ground state  
GS2= np.array([[1,0],[0,0]],dtype = np.complex128)#set exciton ground state#

#Decay Hamiltonians 
HR1= np.array([[1/T11,1/T21],[1/T21,1/T11]],dtype = np.complex128)#set Decay hamiltonian plasmon
HR2= np.array([[1/T12,1/T22],[1/T22,1/T12]],dtype = np.complex128)#set Decay hamiltonian exciton
II = np.eye(2,dtype=np.complex128)# set up an identity matrix 
foa = firstorderaprx
prop = propagate
for j,DTi in enumerate(DT):
    print(j+1,"Out of: ",len(DT))
    Efield = Ef(t,DTi,E0,wl1,wl2,wl3,Pulse,Dt1,Dt2)#generate time delayed Efield
    #initialize the states
    Rho0 = np.asarray([np.copy(GS1) for i in range(Nest)])#initialize plasmon
    Sig0 = np.asarray([np.copy(GS2) for i in range(Nest)])#initialize Exciton 
    for i,E in enumerate(Efield):
        #Iterate plasmon state
        if(np.mod(i,200)==0):
            print(i,"Out of: ",len(Efield))
        E1 = mu1*(E+xi1*(np.sum(Sig0[:,0,1]+Sig0[:,1,0])))  #assign time local efield
        U = np.asarray([foa(E1,E11,E21[i],dt) for i in range(Nest)])       #generate CN Unitary propagator
        Rho0=np.asarray([prop(U[i],HR1,dt,Rho0[i],GS1) for i in range(Nest)])       #Step forward 1 time step
        DR[i,j] = np.sum(Rho0[:,0,0]-Rho0[:,1,1]).real#set excitation ocupation
        R12[i,j]= np.sum(Rho0[:,0,1]).real          #set polarization 
        #Iterate Exciton state
        E2 = mu2*(E+xi2*(np.sum(Rho0[:,0,1]+Rho0[:,1,0])))  #assign time local efield
        U = np.asarray([foa(E2,E12,E22[i],dt)  for i in range(Nest)]) #generate CN Unitary propagator
        Sig0=np.asarray([prop(U[i],HR2,dt,Sig0[i],GS2) for i in range(Nest)])    #Step forward 1 time step
        DS[i,j] = np.sum(Sig0[:,0,0]-Sig0[:,1,1]).real#set excitation ocupation
        S12[i,j]= np.sum(Sig0[:,0,1]).real          #set polarization 

savemat('2photonHeterodyneMC3.mat',{'DS':DS,'S12':S12,'t':t,'DT':DT,
                                     'wl1':wl1,'wl2':wl2,'wlref': wlref,
                                     'T12':T12,'T22':T22})
end = time.time()
print(end-start,' seconds')

