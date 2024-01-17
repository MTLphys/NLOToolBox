# -*- coding: utf-8 -*-
"""
Functions describing the Efield used to excite the coupled exciton plasmon system

@author: mattl
"""
import numpy as np 

def Ef(t,DT,E0,omega1,omega2=0,omega3=0,Epulse=True,Dt1=0,Dt2=0): 
    """An Efield function describing the interaction with the  

    Args:
        t (array or double): the linspace of the time window for the simulation
        DT (double): The width of the pulses
        E0 (double): The strength of the e-field
        omega1 (double):angular frequency of the first pulse 
        omega2 (int, optional): angular frequency of second pulse set to omega 1 if zero. Defaults to 0.
        omega3 (int, optional):  angular frequency of third pulse set to omega 1 if zero. Defaults to 0.
        Epulse (bool, optional): Pulsed(true) or CW mode(false).Defaults to True
        Dt1 (double, optional): delay between pulse 1 and 2. Defaults to 0.
        Dt2 (double, optional): delay between pulse 2 and 3. Defaults to 0.

    Returns:
        _ndarray_: an exciting pulse series 
    """
    Efield = np.zeros(len(t))#setting ups the place holder for the efield values
    omega2 = omega2+(omega2==0)*omega1#Set frequency of second pulse 
    omega3 = omega3+(omega3==0)*omega1#Set frequency of third pulse
    if(Epulse):#choose pulse vs cw
        if((Dt1==0)&(Dt2==0)):#determine if there is a delay between pulse 1(2) and 2(3) 
            Efield= E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1) # set up pulse 1 
        else:
            if((Dt1==0)^(Dt2==0)):# determine if there is just 1 delayed pulse      
                Dt = Dt1+Dt2#set the delay for secondary pulse
                Efield=E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E0*np.sin(omega2*(t+DT))*pulse(t-Dt,5*2*np.pi/omega2)#set up pulse 1 and secondary pulse
            else:
                Efield=E0*np.sin(omega1*(t+DT))*pulse(t,5*2*np.pi/omega1)+E0*np.sin(omega2*(t+DT))*pulse(t-Dt1,5*2*np.pi/omega2)+E0*np.sin(omega3*(t+DT))*pulse(t-Dt2,5*2*np.pi/omega3)#set up all three pulses
    else:
        return E0*np.sin(omega1*(t+DT))#set up cw excitation
    return Efield


def pulse(t,dt):
    """The pulse shape function for a sinusoidal puls shape which goes to zero at either side
    Args: 
        t(array or double): the full time space of the pulse series
        dt(double): the total pulse width
    Returns:
        _ndarray_: A sine pulse envelope""" 
    return np.sin(np.pi*t/dt)**2*np.heaviside(t,.5)*np.heaviside(dt-t,.5)
