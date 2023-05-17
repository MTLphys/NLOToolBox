# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:43:27 2022
Simulation of a stimulated Photon Echo using the NLOToolBox,
This simulation captures all nonlinear lifetime phenomena using 
a MC method in conjuctions with the heterodyning technique to extract
signal. 

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


#assigning the properies of the Exciton 
w2 = np.pi*2.0*.36#gap energy
T1 = 3000.0      #damping constant 
T2 = 1230.0      #Dephasing Constant
mu2 = 1.0         #Efield coupling constants
G2 = .03
E1=0.0           #Ground Energy for EXCITON
E2=np.asarray([E1+w2+G2*np.random.normal()- G2/2 for i in range(Nest)])            #first excited energy For EXCITON
xi2 = 0.00        #System Coupling constant

#generating time space (pulse train and decay time space)
t = np.linspace(0,4000,num=50000) #time 

dt = t[1]-t[0] #time differential 

#set up for the Efield
E0 = .015
wl1 = np.pi*2.0*.36+(np.pi*2.0)*(255.0e-12+99e-9)#first pulse Frequency 
wl2 = np.pi*2.0*.36+(np.pi*2.0)*(1225.0e-12+44e-9)#second Pulse Frequency 
wl3 = np.pi*2.0*.36+(np.pi*2.0)*(555.0e-12+55e-9)#third Pulse Frequency
wlref = np.pi*2.0*.36
Pulse = True#pulse not cw 
Dt1=400 # second pulse @300 fs delay 
Dt2=800 #no first pulse 


wlexp = wl3+wl2-wl1-wlref 
nperiod =np.round(np.pi*2/(wlexp*1.25e7)) #number of points per ~12.5 nsec period
print('number of steps per period:',nperiod)
tstep= np.pi*2/(nperiod*wlexp)# integer step for a ~12.5 nsecond period
print('length of step ',tstep*1e-6,'ns')
cperiod =nperiod*20*tstep #full collection period 
print('total collection time',cperiod*1e-6,'ns')
DT = np.arange(0,cperiod,tstep) #Pulse train
#Data Storage Variables
DS  = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton excited state
S12 = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton polarization

#ground state conditions
GS= np.array([[1,0],[0,0]],dtype = np.complex128)#set exciton ground state#

#Decay Hamiltonians 
HR= np.array([[1/T1,1/T2],[1/T2,1/T1]],dtype = np.complex128)#set Decay hamiltonian exciton
II = np.eye(2,dtype=np.complex128)# set up an identity matrix 
foa = firstorderaprx
prop = propagate
for j,DTi in enumerate(DT):
    print(j+1,"Out of: ",len(DT))
    Efield = Ef(t,DTi,E0,wl1,wl2,wl3,Pulse,Dt1,Dt2)#generate time delayed Efield
    #initialize the states
    Sig0 = np.asarray([np.copy(GS) for i in range(Nest)])#initialize plasmon
    for i,E in enumerate(Efield):
        #Iterate plasmon state
        if(np.mod(i,200)==0):
            print(i,"Out of: ",len(Efield))

        Ef = mu2*(E)  #assign time local efield
        U = np.asarray([foa(Ef,E1,E2[i],dt)  for i in range(Nest)]) #generate CN Unitary propagator
        Sig0=np.asarray([prop(U[i],HR,dt,Sig0[i],GS) for i in range(Nest)])    #Step forward 1 time step
        DS[i,j] = np.sum(Sig0[:,0,0]-Sig0[:,1,1]).real#set excitation ocupation
        S12[i,j]= np.sum(Sig0[:,0,1]).real          #set polarization 

savemat('3photonHeterodyneMC1.mat',{'DS':DS,'S12':S12,'t':t,'DT':DT,
                                     'wl1':wl1,'wl2':wl2,'wl3':wl3,'wlref': wlref,
                                     'T1':T1,'T2':T2})
end = time.time()
print(end-start,' seconds')

