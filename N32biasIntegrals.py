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
            if fullsky:
                L_const = np.sqrt(np.array(L_const)*(np.array(L_const)+1.))
            l_const=np.array(l_const)
            if fullsky:
                l_const=np.sqrt(np.array(l_const)*(np.array(l_const)+1.))
            
            integrand1=spec_int*l_const**3.*np.cos(ang_int)**2.
            integrand2=-spec_int*l_const**2.*L_const*np.cos(ang_int)
            
            phi_integral1+=[simps(integrand1, ang_int)]
            phi_integral2+=[simps(integrand2, ang_int)]

        phi_integral1=np.array(phi_integral1)    
        phi_integral2=np.array(phi_integral2)
        
        if l_max==None:
            int_l1=simps(phi_integral1,np.unique(l_))
            int_l2=simps(phi_integral2,np.unique(l_))
        else:
            index=np.arange(len(np.unique(l_)))
            ind= index[np.where(np.unique(l_)<l_max)]
            int_l1=(simps(phi_integral1[ind],np.unique(l_)[ind]))
            int_l2=(simps(phi_integral2[ind],np.unique(l_)[ind]))

        
        result = np.append(result,int_l1+int_l2)

    return result/(2.*np.pi)**2
    
    
    

def bi_cum(bispec, ell, ang13, bin_size, sample1d, squeezed=False, fullsky=False, l_max=None):
    """ computes integral I0 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    """
        

    print "computing integrated bispectrum..."
    if fullsky:
        print "Using fullsky aproximation"
    # |-L| 
    
    # |l|
    l = ell[2::3]
    
    result = np.array([])
    
    upper_bound=np.int(len(ell)/3/(bin_size))
    bin_size=np.int(bin_size)
    sample1d=np.int(sample1d)


    # for every L interpolate over l and angle and integrate over intrepolated 2d function
    print upper_bound
    for i in np.arange(0,upper_bound):

        i=np.int(i)
        l_ = l[i*bin_size:(i+1)*bin_size]
        ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        phi_integral=[]
        
        
        for j in np.arange(0,sample1d):
            l_const     = l_[j*sample1d:(j+1)*sample1d]
            ang_int     = ang[j*sample1d:(j+1)*sample1d]
            spec_int    = spec[j*sample1d:(j+1)*sample1d]
            
            l_const     = np.array(l_const)
            integrand   = spec_int
            
            phi_integral+=[simps(integrand, ang_int)]
            
    int_l       =(simps(phi_integral,np.unique(l_)))

            
    result = np.append(result,int_l)

    return result
    
    
#def I1(bispec, ell, ang13, bin_size, sample1d, squeezed=False, fullsky=False, l_max=None):
#    """ computes integral I1 eq.(18) in notes 
#    * bispec: array, bispectrum as a function of triangles 
#    * ell   : array of ells that form triangles
#    * ang13 : -\vec L*\vec l/L*l
#    * bin_size: size of sample if one side is held fixed
#    """
#    
#    print "computing beta cross..."
#    if fullsky:
#        print "Using full sky approximation"
#    # |-L| 
#    L = ell[0::3]
#    
#    # |l|
#    l = ell[2::3]
#    
#    upper_bound=np.int(len(ell)/3/(bin_size))
#    bin_size=np.int(bin_size)
#    sample1d=np.int(sample1d)
#    
##    ang13=np.arccos(ang13)
#    
#    result=np.array([])
#    
#    if squeezed:
#        print "Using integration for squeezed configuration"
#        k=0
#        L=np.array(L)
#        assert(len(np.unique(L))<sample1d)
#        l=np.array(l)
#
#        i=np.arange(len(L))
#    # for every L interpolate over l and angle and integrate over intrepolated 2d function
#        while k < len(L):
#            index=i[np.where(L==L[k])]
#            print index
#            L_ = L[index]
#            l_ = l[index]
#            l_ = np.asarray(l_)
#            print min(l_), L[k]
##            print len(np.unique(l_))
#            ang= ang13[index]
#            spec=bispec[index]
#            phi_integral1=[]
#            phi_integral2=[]
#            l_int=[]
#            m=0
#            while m<len(l_):
#                #print l_
#                index=i[np.where(l_==l_[m])]
#                #print l_[m], len(index)
#                l_const = l_[index]
#                ang_int = ang[index]
#                spec_int= spec[index]
#                l_const=np.array(l_const)
#                assert(np.allclose(l_const,l_[m]))
#                
#                if fullsky:
#                    l_const=np.sqrt(l_const*(l_const+1.))
#                
#                integrand1=np.array(spec_int*l_const**2.*l_const*np.sin(ang_int)*np.cos(ang_int)*2.)
#                integrand2=-np.array(spec_int*l_const**2.*L[k]*np.sin(ang_int))
#                
#                phi_integral1+=[simps(integrand1, ang_int)]
#                phi_integral2+=[simps(integrand2, ang_int)]
#                
#                
#                m+=len(l_const)
##            assert(np.allclose(np.unique(np.asarray(l_int)),np.unique(l_)))
#            print len(phi_integral1), len(np.unique(l_))
#            
#            k+=len(L_)
#            int_l1=simps(phi_integral1,np.unique(l_))
#            int_l2=simps(phi_integral2,np.unique(l_))
#
#        
#            result = np.append(result,int_l1+int_l2)
#            
#    else:
#        # for every L interpolate over l and angle and integrate over intrepolated 2d function
#        for i in np.arange(0,upper_bound):
#    
#            i=np.int(i)
#    
#            L_ = L[i*bin_size:(i+1)*bin_size]
#            l_ = l[i*bin_size:(i+1)*bin_size]
#            ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
#            spec=bispec[i*bin_size:(i+1)*bin_size]
#            phi_integral1=[]
#            phi_integral2=[]
#            
#            
#            for j in np.arange(0,sample1d):
#                l_const = l_[j*sample1d:(j+1)*sample1d]
#                L_const = L_[j*sample1d:(j+1)*sample1d]
#                ang_int = ang[j*sample1d:(j+1)*sample1d]
#                spec_int= spec[j*sample1d:(j+1)*sample1d]
#                l_const=np.array(l_const)
#                if fullsky:
#                    l_const=np.sqrt(np.array(l_const)*(np.array(l_const)+1.))
#                    L_const=np.sqrt(np.array(L_const)*(np.array(L_const)+1.))
#        
#                integrand1=np.array(spec_int*l_const**2.*l_const*np.sin(ang_int)*np.cos(ang_int)*2.)
#                integrand2=-np.array(spec_int*l_const**2.*L_const*np.sin(ang_int))
#                
#                phi_integral1+=[simps(integrand1, ang_int)]
#                phi_integral2+=[simps(integrand2, ang_int)]
#    
#            phi_integral1=np.array(phi_integral1)    
#            phi_integral2=np.array(phi_integral2)
#            if l_max==None:
#                int_l1=(simps(phi_integral1,np.unique(l_)))
#                int_l2=simps(phi_integral2,np.unique(l_))
#            else:
#                index=np.arange(len(np.unique(l_)))
#                ind= index[np.where(np.unique(l_)<l_max)]
#                int_l1=(simps(phi_integral1[ind],np.unique(l_)[ind]))
#                int_l2=(simps(phi_integral2[ind],np.unique(l_)[ind]))
#           
#            
#
#        
#            result = np.append(result,int_l1+int_l2)
#
#    return result/(2.*np.pi)**2  #times two since we only cover upper half of plane