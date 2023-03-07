# -*- coding: utf-8 -*-
"""
Functions describing the Efield used to excite the coupled exciton plasmon system

@author: mattl
"""
import numpy as np 

def Ef(t,DT,E0,omega1,omega2=0,omega3=0,Epulse=True,Dt1=0,Dt2=0): 
    Efield = np.zeros(len(t))
    omega2 = omega2+(omega2==0)*omega1 
    omega3 = omega3+(omega3==0)*omega1
    if(Epulse):
        if((Dt1==0)&(Dt2==0)):
            Efield= E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)
        else:
            if((Dt1==0)^(Dt2==0)):    
                Dt = Dt1+Dt2
                Efield=E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E0*np.sin(omega2*(t+DT))*pulse(t-Dt,5*2*np.pi/omega2)
            else:
                Efield=E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E0*np.sin(omega2*(t+DT))*pulse(t-Dt1,5*2*np.pi/omega2)+E0*np.sin(omega3*(t+DT))*pulse(t-Dt2,5*2*np.pi/omega3)
    else:
        return E0*np.sin(omega1*(t+DT))
    return Efield
def Efth(t,DT,theta0,omega1,omega2=0,omega3=0,Epulse=True,Dt1=0,Dt2=0,theta1=0,theta2=0): 
    Efield = np.zeros(len(t))
    a = 5*2*np.pi/omega1
    E1 = theta0*a/np.pi 
    E2 = theta1*a/np.pi 
    E3 = theta2*a/np.pi 
    omega2 = omega2+(omega2==0)*omega1 
    omega3 = omega3+(omega3==0)*omega1
    if(Epulse):
        if((Dt1==0)&(Dt2==0)):
            Efield= E1*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)
        else:
            if((Dt1==0)^(Dt2==0)):    
                Dt = Dt1+Dt2
                Efield=E1*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E2*np.sin(omega2*(t+DT))*pulse(t-Dt,5*2*np.pi/omega2)
            else:
                Efield=E1*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E2*np.sin(omega2*(t+DT))*pulse(t-Dt1,5*2*np.pi/omega2)+E3*np.sin(omega3*(t+DT))*pulse(t-Dt2,5*2*np.pi/omega3)
    else:
        return E1*np.sin(omega1*(t+DT))
    return Efield
def MPulse(t,E0,omega1,Dt=0,Phi=0): 
    #Efield combinded function 
    Efield = np.zeros(len(t))
    Efield= E0*np.sin(omega1*t-Dt-Phi)*pulse(t-Dt,5*2*np.pi/omega1)
    return Efield 
def pulse(t,dt):
    #pulse envelope 
    return np.sin(np.pi*t/dt)**2*np.heaviside(t,.5)*np.heaviside(dt-t,.5)
