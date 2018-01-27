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


            integrand=spec_int*np.sin(theta)**2

            l_integral+=[simps(integrand, theta)]


        int_ang=simps(l_integral*l**3,l)
        result = np.append(result,int_ang)

    return result/(2.*np.pi)**2



def I2(bispec, L, l, theta, len_l, len_L,len_ang):

    result = np.array([])
    print len_l, len_L
    bin_size = len_ang*len_l
    plt.figure()
    for i in np.arange(0,len_L):

        i=np.int(i)
        L_=L[i]
        spec=bispec[i*bin_size:(i+1)*bin_size]
        l_integral=[]


        for j in np.arange(0,len_l):
            spec_int  = spec[j*len_ang:(j+1)*len_ang]
            l_=l[j]
            integrand = spec_int*np.cos(theta)*(l_*np.cos(theta)-L_)

            l_integral+=[simps(integrand, theta)]


        int_ang=simps(l_integral*l**2,l)
        if i in [54,55]:
            plt.semilogx(l,l_integral*l**2,'ro',label='%d'%(L[i]))
            plt.legend(loc='best',ncol=3)
        result = np.append(result,int_ang)
    plt.savefig('check_I2_integrands_b.pdf',bbox_inches='tight')
    return result/(2.*np.pi)**2


def filt(l,FWHM):

    FWHM_ = FWHM/60.*(np.pi/180.)
    sigma2= FWHM_**2/8./np.log(2.)

    return np.exp(-l**2*sigma2/2.)

def skew(bispec, FWHM, L, l, Ll, theta, len_l, len_L,len_ang, kappa):

    result = np.array([])
    bin_size = len_ang*len_l

    for i in np.arange(0,len_L):
        L_=L[i]
        i=np.int(i)
        spec=bispec[i*bin_size:(i+1)*bin_size]
        Ll_=Ll[i*bin_size:(i+1)*bin_size]

        l_integral=[]
        for j in np.arange(0,len_l):
            l_=l[j]
            spec_int  = spec[j*len_ang:(j+1)*len_ang]
            l3 = Ll_[j*len_ang:(j+1)*len_ang]
            integrand = spec_int*filt(L_,FWHM)*filt(l_,FWHM)*filt(l3,FWHM)
            if kappa:
                integrand*=1./8.*(L_*l_*l3)**2

            l_integral+=[simps(integrand, theta)]


        int_ang=simps(l_integral*l,l)

        result = np.append(result,int_ang)

    res=simps(result*L,L)

    return res/(2.*np.pi)**3