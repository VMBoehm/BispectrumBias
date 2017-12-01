# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:05:20 2016

@author: Vboehm
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

def beta_RR(bispec, ell, ang13, len_L, len_l, len_ang, R, fullsky=False):
    """ computes integral beta_RR, Appendix D in the notes
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    * R :smoothing scale
    """
    
    print "computing betaRR with R=%f "%R
    if fullsky:
        print "using fullsky approximation"
    # |-L| 
    L = ell[0::3]
    # |L-l|
    Ll = ell[1::3]
    # |l|
    l = ell[2::3]
    
    result = np.array([])
    
    bin_size=len_ang*len_l

    for i in np.arange(0,len_L):

        i=np.int(i)
        
        L_  = L[i*bin_size:(i+1)*bin_size]
        l_  = l[i*bin_size:(i+1)*bin_size]
        ang = ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        Ll_ = Ll[i*bin_size:(i+1)*bin_size]
        spec=bispec[i*bin_size:(i+1)*bin_size]
        
        phi_integral=[]
        
        for j in np.arange(0,len_l):
            l_const = l_[j*len_ang:(j+1)*len_ang]
            ang_int = ang[j*len_ang:(j+1)*len_ang]
            Ll_int = Ll_[j*len_ang:(j+1)*len_ang]
            spec_int= spec[j*len_ang:(j+1)*len_ang]
            L_const = L_[j*len_ang:(j+1)*len_ang]
            
            if fullsky:
                L_const = np.sqrt(np.array(L_const)*(np.array(L_const)+1.))
            
            l_const=np.array(l_const)
            
            integrand=spec_int*l_const*np.exp(-(l_const*R)**2/2)*np.exp(-(Ll_int*R)**2/2)
            
            phi_integral+=[simps(integrand1, ang_int)]
     
        result=(simps(phi_integral1,np.unique(l_)))
       
return result/((2*np.pi)**2)