# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:43:27 2022
Simulation of Homogeneously broadened 4WM using the NLOToolBox,
This simulation captures homogeneous lifetime phenomena using 
a heterodyning technique to extract FWM frequency signal.

Use the program  
@author: mattl
"""
#import packages
import numpy as np #import Math resources 
from scipy.io import savemat #import IO 
from NLOToolBox.Efield import Ef #import user defined Efield function
from NLOToolBox.Propagation import firstorderaprx, propagate #import user defined Propagator functions 
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt 
#Moving on to the coupled system simulation 
import time
#set up a timer 
start = time.time()#Timing Code execution 
Nest = 200
def sfilt(t,fwhm):
    return (1/np.cosh(np.log(3+np.sqrt(8))*((t-np.max(t)/2)/2)/fwhm))**2
def welchwin(j,n):
    return 1.0 -(2.0*j/(n-1.0)-1.0)**2

#assigning the properies of the Exciton 
w2 = np.pi*2.0*.36#gap energy
T1 = 3000.0      #damping constant 
T2 = 250.0      #Dephasing Constant
mu2 = 1.0         #Efield coupling constants
G2 = .03
E12=0.0           #Ground Energy for EXCITON
E22=E12+w2          #first excited energy For EXCITON
xi2 = 0.00        #System Coupling constant

#generating time space (pulse train and decay time space)
t = np.linspace(0,4000,num=50000) #time 

dt = t[1]-t[0] #time differential 

#set up for the Efield
E0 = .015
wl1 = np.pi*2.0*.36+(np.pi*2.0)*(255.0e-12+12e-9)#first pulse Frequency 
wl2 = np.pi*2.0*.36+(np.pi*2.0)*(1225.0e-12+7e-9)#second Pulse Frequency 
wl3 = np.pi*2.0*.36+(np.pi*2.0)*(655.0e-12+5e-9)#third Pulse Frequency
wlref = np.pi*2.0*.36
Pulse = True#pulse not cw 
Dt1=200 # second pulse @300 fs delay 
Dt2=np.linspace(0,4000,num=Nest)  


wlexp = wl3+wl2-wl1-wlref 
nperiod =np.round(np.pi*2/(wlexp*1.25e7)) #number of points per ~12.5 nsec period
print('number of steps per period:',nperiod)
tstep= np.pi*2/(nperiod*wlexp)# integer step for a ~12.5 nsecond period
print('length of step ',tstep*1e-6,'ns')
cperiod =nperiod*12*tstep #full collection period 
print('total collection time',cperiod*1e-6,'ns')
steps = np.round(cperiod/tstep)
print('total collection time steps',steps,'steps')
DT = np.arange(0,cperiod,tstep) #Pulse train
#Data Storage Variables
DS  = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton excited state
S12 = np.zeros((len(t),len(DT)),dtype=np.complex128)#initialize exciton polarization

#ground state conditions
GS2= np.array([[1,0],[0,0]],dtype = np.complex128)#set exciton ground state#

#Decay Hamiltonians 
HR2= np.array([[1/T1,1/T2],[1/T2,1/T1]],dtype = np.complex128)#set Decay hamiltonian exciton
II = np.eye(2,dtype=np.complex128)# set up an identity matrix 
foa = firstorderaprx
prop = propagate
Smax = 1
fw = 8
lt = np.zeros((len(t),Nest),dtype=np.complex128)
kernal = sfilt(t,fw)
kernal = kernal/np.sum(kernal)
N = len(DT)
jj = np.array([i for i in range(N)])
f_H = np.exp(1j*wlref*DT) 
for k in range(Nest):
    print(k,"Out of: ",Nest)
    for j,DTi in enumerate(DT):
        Efield = Ef(t,DTi,E0,wl1,wl2,wl3,Pulse,Dt1,Dt2[k]) #generate time delayed Efield
        #initialize the states
        print(j,"Out of:",steps)
        Sig0 = np.copy(GS2) 
        for i,E in enumerate(Efield):
            #Iterate plasmon state
            #set polarization 
            #Iterate Exciton state
            E2 = mu2*(E)  #assign time local efield
            U = foa(E2,E12,E22,dt)  #generate CN Unitary propagator
            Sig0=prop(U,HR2,dt,Sig0,GS2)     #Step forward 1 time step
            DS[i,j] = (Sig0[0,0]-Sig0[1,1]).real#set excitation ocupation
            S12[i,j]= (Sig0[0,1]).imag          #set polarization 
    if(k==0):
        Smax =np.max(S12)
    
    Hsig = np.asarray([fftconvolve(np.abs(S12[:,ii]+Smax*np.sin(wlref*(t+DT[ii])))**2-np.abs(S12[:,ii]-Smax*np.sin(wlref*(t+DT[ii])))**2,kernal,'same') for ii in range(len(DT))])
    #Do discrete fourier transformation for just the desired frequency
    lt[:,k]= np.asarray([np.abs(np.mean(f_H*(welchwin(jj, N)*(Hsig[:,j]-np.mean(Hsig[:,j]))))) for j in range(len(t))])
    #Mix with heterodyne signal 
    plt.plot(t,np.abs(lt[:,k]))
    plt.yscale('log')
    plt.ylim(1e-10,1)
    plt.show()
    
savemat('3photonHeterodyneDL0.mat',{'DS':DS,'S12':S12,'lt':lt,'t':t,'DT':DT,
                                     'wl1':wl1,'wl2':wl2,'wl3':wl3,'wlref': wlref,
                                     'T1':T1,'T2':T2,'Dt2':Dt2,'Nest':Nest})
end = time.time()
print(end-start,' seconds')

