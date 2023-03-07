# -*- coding: utf-8 -*-
"""
The process file for the Stimulated Photon Echo Simulation, Make sure to match the Load mat file name with the 
out put of the generation file

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
File = loadmat('3photonHeterodyneMC4.mat')
S12= File['S12']
DT = File['DT'][0]
t = File['t'][0]
wl1= File['wl1'][0]
wl2= File['wl2'][0]
wl3 =np.pi*2.0*(.36+655.0e-12+3.12e-9)
T1 = File['T1'][0]
T2 = File['T2'][0]
sig = wl1*10
N = len(DT)
jj = np.array([i for i in range(N)])
wref = np.pi*2.0*.36

wlexp = wl3+wl2-wl1-wref 
wrabi = np.pi*2.0*.36
kernal = sfilt(t,fw)
kernal = kernal/np.sum(kernal)
Smax =np.max(S12)
#Mix with heterodyne signal 
Hsig = np.asarray([fftconvolve(np.abs(S12[:,i]+Smax*np.sin(wref*(t+DT[i])))**2,kernal,'same') for i in range(len(DT))])


#Do discrete fourier transformation for just the desired frequency
wH = wlexp
f_H = np.exp(1j*wH*DT) 
lt= np.asarray([np.abs(np.mean(f_H*(welchwin(jj, N)*(Hsig[:,j]-np.mean(Hsig[:,j]))))) for j in range(len(t))])
avgsig= np.asarray([np.abs(np.mean(Hsig[:,j])) for j in range(len(t))])



plt.title('Time Dependent Heterodyned Polarization ')
plt.xlabel('Delay[fs]')
plt.ylabel('Pulse x [12500ps]')
plt.imshow(Hsig,extent=[0,np.max(t),0,np.max(DT)/1.25e7],aspect =5,vmax=5*np.mean(Hsig),cmap='jet')
plt.show()
plt.title('Oscillation Spectrum excited State Occupation at 600 fs over 168 Pulses')
plt.xlabel('Frequency[Mhz]')
plt.ylabel('Intenstiy[arb. units]')
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(DT),DT[1]))*1e9,np.fft.fftshift(np.abs(np.fft.fft(welchwin(jj, N)*(Hsig[:,10000]-np.mean(Hsig[:,10000]))))),label = 'PL Mixing FFT',marker='o')
plt.axvline(x =np.abs((wlexp)*1e9/(2*np.pi)),color = 'k',label='2*k2 - k1- kref')
plt.axvline(x =np.abs((wl1-wl2)*1e9/(2*np.pi)),color = 'r',label='k1 - k2')
#plt.axvline(x =np.abs((wrab-wl1)))
plt.xlim(0,4*wlexp*1e9/2/np.pi)
#plt.ylim(0,1e-3)
plt.legend()
plt.show()

plt.title('Heterodyne vs T2 Decay of Polarization Signal')
plt.plot(t,lt/np.max(lt[t>500]),label ='2*k2-k1-kref Frequency signal')
plt.ylabel('Intensity [arb. units.]')
plt.xlabel('Real Time Delay [fs]')
#plt.plot(t,avgsig/np.max(avgsig[t>200]),label='mean signal over 320 pulses')
#plt.plot(t,(np.abs(S12[:,1]))/np.max(np.abs(S12[t>200])),label='rawsignal')
#plt.yscale('log')
plt.ylim(1e-12,2)
#kernal_size = 1
#kernal = np.ones(kernal_size)/kernal_size
#smooth = np.convolve(lt,kernal,mode='same')
plt.plot(t,np.exp(-(t/T22))/np.max((np.exp(-(t/T22)))[t>500]),label='T2 Decay')
plt.axvline(x =0+5*np.pi/wl1,color = 'orange',label='Pulse 1')
plt.axvline(x =0+5*np.pi/wl1+600,color = 'red',label='Pulse 2')
plt.axvline(x =0+10*np.pi/wl1+1200,color = 'black',label='echo')
plt.legend()
