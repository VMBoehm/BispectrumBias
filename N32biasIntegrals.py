# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:08:51 2015

@author: Vanessa M. Boehm
"""
from __future__ import division
import numpy as np
import pylab as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from scipy.integrate import simps
#import scipy.interpolate as interpolate

#import HelperFunctions as hf
import warnings
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings('error')
#    
def I0(bispec, ell, ang13, len_L, len_l, len_ang, fullsky=False, l_max=None):
    """ computes integral I0 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    """
        
    print "computing beta perp..."
    if fullsky:
        print "Using fullsky aproximation"

    l = ell[2::3]
    
    result = np.array([])
    
    bin_size = len_ang*len_l
    
    for i in np.arange(0,len_L):

        i=np.int(i)
        l_ = l[i*bin_size:(i+1)*bin_size]
        ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        phi_integral=[]
        
        
        for j in np.arange(0,len_l):
            l_const = l_[j*len_ang:(j+1)*len_ang]
            ang_int = ang[j*len_ang:(j+1)*len_ang]
            spec_int= spec[j*len_ang:(j+1)*len_ang]
            
            l_const=np.array(l_const)
            if fullsky:
                l_const=np.sqrt(np.array(l_const)*(np.array(l_const)+1.))
            integrand=spec_int*l_const**3*np.sin(ang_int)**2
            phi_integral+=[simps(integrand, ang_int)]
        
        phi_integral=np.array(phi_integral)
        if l_max==None:
            int_l=(simps(phi_integral,np.unique(l_)))
        else:
            index=np.arange(len(np.unique(l_)),dtype=int)
            ind= index[np.where(np.unique(l_)<l_max)]
            int_l=(simps(phi_integral[ind],np.unique(l_)[ind]))

        
        result = np.append(result,int_l)

    return result/(2.*np.pi)**2

    
            
def I2(bispec, ell, ang13, len_L,len_l,len_ang, fullsky=False, l_max=None):
    """ computes integral I0 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    """
    
    print "computing beta parallel..."
    if fullsky:
        print "using fullsky approximation"
    # |-L| 
    L = ell[0::3]
    
    # |l|
    l = ell[2::3]
    
    result = np.array([])

    bin_size = len_l*len_ang    
    
    for i in np.arange(0,len_L):

        i=np.int(i)
        
        L_ = L[i*bin_size:(i+1)*bin_size]
        l_ = l[i*bin_size:(i+1)*bin_size]
        ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        phi_integral1=[]
        phi_integral2=[]
                
        for j in np.arange(0,len_l):
            l_const = l_[j*len_ang:(j+1)*len_ang]
            ang_int = ang[j*len_ang:(j+1)*len_ang]
            spec_int= spec[j*len_ang:(j+1)*len_ang]
            L_const = L_[j*len_ang:(j+1)*len_ang]
            l_const=np.array(l_const)

            #ldl l cos(lcos-L) B
            integrand =spec_int*np.cos(ang_int)**2
            integrand2=spec_int*np.cos(ang_int)*L_const
            phi_integral1+=[simps(integrand, ang_int)]
            phi_integral2+=[simps(integrand2, ang_int)]
        
        ll = np.unique(l_)
        phi_integral1=np.array(phi_integral1)*ll**3
        phi_integral2=np.array(phi_integral2)*ll**2
        
        if i in [5,10,20,30,40,50,60,70,80,90]:
            plt.figure()
            plt.plot(ll,phi_integral1,ls='',marker='o',label=L_const[0])
            plt.plot(ll,phi_integral2,ls='',marker='o')
            plt.plot(ll,phi_integral2+phi_integral1,ls='',marker='o')
            plt.legend(loc='best')
            plt.xlim(L_const[0]-200,L_const[0]+200)
            plt.savefig('phi_integrals%d.png'%i)
            
            
        if l_max==None:
            int_l=simps(phi_integral1,ll)
            int_l2=simps(phi_integral2,ll)
        else:
            index=np.arange(len(ll))
            ind= index[np.where(ll<l_max)]
            int_l=(simps(phi_integral1[ind],ll[ind]))
        
        result = np.append(result,int_l-int_l2)

    return result/(2.*np.pi)**2
    
    
    

def bi_cum(bispec, ell, ang13, len_L, len_l, len_ang, fullsky=False, l_max=None):
    """ computes integral I0 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    """
        

    print "computing beta perp..."
    if fullsky:
        print "Using fullsky aproximation"

    l = ell[2::3]
    
    result = np.array([])
    
    bin_size = len_ang*len_l
    
    for i in np.arange(0,len_L):

        i=np.int(i)
        l_ = l[i*bin_size:(i+1)*bin_size]
        ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        phi_integral=[]
        
        
        for j in np.arange(0,len_l):
            l_const = l_[j*len_ang:(j+1)*len_ang]
            ang_int = ang[j*len_ang:(j+1)*len_ang]
            spec_int= spec[j*len_ang:(j+1)*len_ang]
            
            l_const=np.array(l_const)
            integrand=spec_int*l_const
            phi_integral+=[simps(integrand, ang_int)]
        
        phi_integral=np.array(phi_integral)

        int_l=simps(phi_integral,np.unique(l_))
        
        result = np.append(result,int_l)

    return result
