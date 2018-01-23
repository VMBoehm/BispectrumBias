# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:08:51 2015

@author: Vanessa M. Boehm
"""
from __future__ import division
import numpy as np
import pylab as plt
from scipy.integrate import simps


def I0(bispec, L, l, theta, len_l, len_L,len_ang):

    result = np.array([])

    bin_size = len_ang*len_l

    for i in np.arange(0,len_L):

        i=np.int(i)
        spec=bispec[i*bin_size:(i+1)*bin_size]
        l_integral=[]


        for j in np.arange(0,len_l):
            spec_int  = spec[j*len_ang:(j+1)*len_ang]


            integrand=spec_int*l**3*np.sin(theta)**2

            l_integral+=[simps(integrand, theta)]


        int_ang=simps(l_integral,l)
        result = np.append(result,int_ang)

    return result/(2.*np.pi)**2



def I2(bispec, L, l, theta, len_l, len_L,len_ang):

    result = np.array([])

    bin_size = len_ang*len_l

    for i in np.arange(0,len_L):

        i=np.int(i)
        L_=L[i]
        spec=bispec[i*bin_size:(i+1)*bin_size]
        l_integral=[]


        for j in np.arange(0,len_l):
            spec_int  = spec[j*len_ang:(j+1)*len_ang]


            integrand = spec_int*l**2*np.cos(theta)*(l*np.cos(theta)-L_)

            l_integral+=[simps(integrand, theta)]


        int_ang=simps(l_integral,l)
        result = np.append(result,int_ang)

    return result/(2.*np.pi)**2