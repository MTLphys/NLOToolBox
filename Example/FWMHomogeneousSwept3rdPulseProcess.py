# -*- coding: utf-8 -*-
"""
Program used to process the out put of the FWMHomogeneousSwept3rdPulse file make sure to match the load mat 
file name with that created by the generating program

@author: mattl
"""
import matplotlib.pyplot as plt
import numpy as np 
from scipy.io import loadmat
from scipy.signal import fftconvolve


def sfilt(t,fwhm):
    return (1/np.cosh(np.log(3+np.sqrt(8))*((t-np.max(t)/2)/2)/fwhm))**2
def welchwin(j,n):
    return 1.0 -(2.0*j/(n-1.0)-1.0)**2

fw = 8.0
E0 = .015
File = loadmat('3photonHeterodyneDL0.mat')
Hsig= File['lt']
DT = File['DT'][0]
t = File['t'][0]
wl1= File['wl1'][0]
wl2= File['wl2'][0]
wl3 =File['wl3'][0]
T12 = File['T12'][0]
T22 = File['T22'][0]
Nest = 200
Dt2=np.linspace(0,4000,num=Nest) 
sig = wl1*10
N = len(DT)
jj = np.array([i for i in range(N)])
wref = np.pi*2.0*.36

wlexp = wl3+wl2-wl1-wref 
wrabi = np.pi*2.0*.36
kernal = sfilt(t,fw)
kernal = kernal/np.sum(kernal)
#Mix with heterodyne signal 
HSigSmooth = np.asarray([fftconvolve(Hsig[:,i],kernal,'same') for i in range(Nest)])

plt.title('Heterodyne Series, Varied Delays')
tsig = np.asarray([np.sum(HSigSmooth[i,:]) for i in range(Nest)])
plt.plot(Dt2,tsig-2.843)
plt.yscale('log')
#plt.imshow(np.abs(HSigSmooth),aspect=50000/200)
plt.ylabel('Intensity [arb. units.]')
plt.xlabel('Real Time Delay [fs]')
#plt.plot(t,avgsig/np.max(avgsig[t>200]),label='mean signal over 320 pulses')
#plt.plot(t,(np.abs(S12[:,1]))/np.max(np.abs(S12[t>200])),label='rawsignal')
#plt.yscale('log')

#kernal_size = 1
#kernal = np.ones(kernal_size)/kernal_size
#smooth = np.convolve(lt,kernal,mode='same')
#plt.plot(t,np.exp(-(t/T22))/np.max((np.exp(-(t/T22)))[t>500]),label='T2 Decay')
#plt.axvline(x =0+5*np.pi/wl1,color = 'orange',label='Pulse 1')
#plt.axvline(x =0+5*np.pi/wl1+600,color = 'red',label='Pulse 2')
#plt.axvline(x =0+10*np.pi/wl1+1200,color = 'black',label='echo')
#Xplt.legend()
