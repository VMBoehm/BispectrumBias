# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:08:51 2015

@author: Vanessa M. Boehm
"""
from __future__ import division
import numpy as np
from scipy.integrate import simps

def integrate_bispec(bispec, ell, ang13, len_L, len_ang, fullsky=False):
    """ computes integral I0 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * len_L : length of unique L and l values
    * len_ang: length of unique angles
    """
        
    print "computing beta perp..."
    if fullsky:
        print "Using L*(L+1) instead of L^2"
    # |-L| 
    L = ell[0::3]
    
    # |l|
    l = ell[2::3]
    
    result = np.zeros((5,len_L))
    
    bin_size = int(len_L*len_ang)
    upper    = int(len(L)/bin_size)
    print upper, bin_size

    for i in np.arange(0,upper):
        
        L_  = L[i*bin_size:(i+1)*bin_size]
        l_  = l[i*bin_size:(i+1)*bin_size]
        ang = ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        phi_integral=np.zeros((5,len_L))

        
        ## angular integration
        for j in np.arange(0,len_L):
            l_const = l_[j*len_ang:(j+1)*len_ang]

            ang_int = ang[j*len_ang:(j+1)*len_ang]
            spec_int= spec[j*len_ang:(j+1)*len_ang]
            l_const=np.array(l_const)
            if fullsky:
                #derivatives of spherical harmonics...
                l_const=np.sqrt(np.array(l_const)*(np.array(l_const)+1.))
            integrand=spec_int*l_const #d^2l-> l*dl dtheta
            integrandIa=integrand*l_const*np.cos(ang_int)
            integrandIb=integrand*l_const*np.cos(ang_int)*np.sin(ang_int)
            integrandIIa=integrand*l_const**2*np.cos(ang_int)**2
            integrandIIb=integrand*l_const**2*np.sin(ang_int)**2
            integrandIIc=integrand*l_const**2*np.cos(ang_int)*np.sin(ang_int)
            phi_integral[0][j]=simps(integrandIa, ang_int)
            phi_integral[1][j]=simps(integrandIb, ang_int)
            phi_integral[2][j]=simps(integrandIIa, ang_int)
            phi_integral[3][j]=simps(integrandIIb, ang_int)
            phi_integral[4][j]=simps(integrandIIc, ang_int)
            
        for m in range(5):
            result[m][i]  = simps(phi_integral[m],np.unique(l_))      
    
    return result/(2.*np.pi)**2
  

    
    
def I1(bispec, ell, ang13, bin_size, sample1d, squeezed=False, fullsky=False, l_max=None):
    """ computes integral I1 eq.(18) in notes 
    * bispec: array, bispectrum as a function of triangles 
    * ell   : array of ells that form triangles
    * ang13 : -\vec L*\vec l/L*l
    * bin_size: size of sample if one side is held fixed
    """
    
    print "computing beta cross..."
    if fullsky:
        print "Using full sky approximation"
    # |-L| 
    L = ell[0::3]
    
    # |l|
    l = ell[2::3]
    
    upper_bound=np.int(len(ell)/3/(bin_size))
    bin_size=np.int(bin_size)
    sample1d=np.int(sample1d)
    
#    ang13=np.arccos(ang13)
    
    result=np.array([])
    
    if squeezed:
        print "Using integration for squeezed configuration"
        k=0
        L=np.array(L)
        assert(len(np.unique(L))<sample1d)
        l=np.array(l)

        i=np.arange(len(L))
    # for every L interpolate over l and angle and integrate over intrepolated 2d function
        while k < len(L):
            index=i[np.where(L==L[k])]
            print index
            L_ = L[index]
            l_ = l[index]
            l_ = np.asarray(l_)
            print min(l_), L[k]
#            print len(np.unique(l_))
            ang= ang13[index]
            spec=bispec[index]
            phi_integral1=[]
            phi_integral2=[]
            l_int=[]
            m=0
            while m<len(l_):
                #print l_
                index=i[np.where(l_==l_[m])]
                #print l_[m], len(index)
                l_const = l_[index]
                ang_int = ang[index]
                spec_int= spec[index]
                l_const=np.array(l_const)
                assert(np.allclose(l_const,l_[m]))
                
                if fullsky:
                    l_const=np.sqrt(l_const*(l_const+1.))
                
                integrand1=np.array(spec_int*l_const**2.*l_const*np.sin(ang_int)*np.cos(ang_int)*2.)
                integrand2=-np.array(spec_int*l_const**2.*L[k]*np.sin(ang_int))
                
                phi_integral1+=[simps(integrand1, ang_int)]
                phi_integral2+=[simps(integrand2, ang_int)]
                
                
                m+=len(l_const)
#            assert(np.allclose(np.unique(np.asarray(l_int)),np.unique(l_)))
            print len(phi_integral1), len(np.unique(l_))
            
            k+=len(L_)
            int_l1=simps(phi_integral1,np.unique(l_))
            int_l2=simps(phi_integral2,np.unique(l_))

        
            result = np.append(result,int_l1+int_l2)
            
    else:
        # for every L interpolate over l and angle and integrate over intrepolated 2d function
        for i in np.arange(0,upper_bound):
    
            i=np.int(i)
    
            L_ = L[i*bin_size:(i+1)*bin_size]
            l_ = l[i*bin_size:(i+1)*bin_size]
            ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
            spec=bispec[i*bin_size:(i+1)*bin_size]
            phi_integral1=[]
            phi_integral2=[]
            
            
            for j in np.arange(0,sample1d):
                l_const = l_[j*sample1d:(j+1)*sample1d]
                L_const = L_[j*sample1d:(j+1)*sample1d]
                ang_int = ang[j*sample1d:(j+1)*sample1d]
                spec_int= spec[j*sample1d:(j+1)*sample1d]
                l_const=np.array(l_const)
                if fullsky:
                    l_const=np.sqrt(np.array(l_const)*(np.array(l_const)+1.))
                    L_const=np.sqrt(np.array(L_const)*(np.array(L_const)+1.))
        
                integrand1=np.array(spec_int*l_const**2.*l_const*np.sin(ang_int)*np.cos(ang_int)*2.)
                integrand2=-np.array(spec_int*l_const**2.*L_const*np.sin(ang_int))
                
                phi_integral1+=[simps(integrand1, ang_int)]
                phi_integral2+=[simps(integrand2, ang_int)]
    
            phi_integral1=np.array(phi_integral1)    
            phi_integral2=np.array(phi_integral2)
            if l_max==None:
                int_l1=(simps(phi_integral1,np.unique(l_)))
                int_l2=simps(phi_integral2,np.unique(l_))
            else:
                index=np.arange(len(np.unique(l_)))
                ind= index[np.where(np.unique(l_)<l_max)]
                int_l1=(simps(phi_integral1[ind],np.unique(l_)[ind]))
                int_l2=(simps(phi_integral2[ind],np.unique(l_)[ind]))
           
            

        
            result = np.append(result,int_l1+int_l2)

    return result/(2.*np.pi)**2  #times two since we only cover upper half of plane
    
            
def I2(bispec, ell, ang13, bin_size, sample1d, squeezed=False, fullsky=False, l_max=None):
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
    
    upper_bound=np.int(len(ell)/3/(bin_size))
    bin_size=np.int(bin_size)
    sample1d=np.int(sample1d)

    if squeezed:
        L=np.array(L)
        l=np.array(l)
        upper_bound=len(np.unique(L))
        print upper_bound
        print "Using integration for squeezed configuration"
        k=0
        count=0
    # for every L interpolate over l and angle and integrate over intrepolated 2d function
        while k < len(L):
            i=np.arange(len(L))
            index=i[np.where(L==L[k])]
            L_ = L[index]
            l_ = l[index]
            ang= ang13[index]
            spec=bispec[index]
            phi_integral1=[]
            phi_integral2=[]
            l_int=[]
            m=0
            while m<len(l_):
                index=i[np.where(l_==l_[m])]
                l_const = l_[index]
                ang_int = ang[index]
                spec_int= spec[index]
                L_const=L_[index]
    
                integrand1=spec_int*l_const**3*np.cos(ang_int)**2
                integrand2=-spec_int*l_const**2*L_const*np.cos(ang_int)
                
                phi_integral1+=[simps(integrand1, ang_int)]
                phi_integral2+=[simps(integrand2, ang_int)]
                l_int+=[l_const]
                m+=len(l_const)
            
            
#            print len(np.array(phi_integral))
#            if count<5:
#                n=6
#                p=0
#            elif (count<upper_bound-100 and count>=5):
#                p=0
#                n=i+1
#            else:
#                p=0
#                n=upper_bound-100
            

#            int_l=(simps(phi_integral[p:n],np.unique(l_)[p:n])+simps(phi_integral[n:n+80],np.unique(l_)[n:n+80])+simps(phi_integral[n+80:-1],np.unique(l_)[n+80:-1]))

            int_l1=simps(phi_integral1,np.unique(l_))
            int_l2=simps(phi_integral2,np.unique(l_))
            k+=len(L_)
            count+=1
    
            result = np.append(result,int_l1+int_l2)
    else:
    # for every L interpolate over l and angle and integrate over intrepolated 2d function
        print upper_bound
        for i in np.arange(0,upper_bound):
    
            i=np.int(i)
            
            L_ = L[i*bin_size:(i+1)*bin_size]
            l_ = l[i*bin_size:(i+1)*bin_size]
            ang= ang13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
            spec=bispec[i*bin_size:(i+1)*bin_size]
            phi_integral1=[]
            phi_integral2=[]
            
            
            for j in np.arange(0,sample1d):
                l_const = l_[j*sample1d:(j+1)*sample1d]
                ang_int = ang[j*sample1d:(j+1)*sample1d]
                spec_int= spec[j*sample1d:(j+1)*sample1d]
                L_const = L_[j*sample1d:(j+1)*sample1d]
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
                int_l1=(simps(phi_integral1,np.unique(l_)))
                int_l2=simps(phi_integral2,np.unique(l_))
            else:
                index=np.arange(len(np.unique(l_)))
                ind= index[np.where(np.unique(l_)<l_max)]
                int_l1=(simps(phi_integral1[ind],np.unique(l_)[ind]))
                int_l2=(simps(phi_integral2[ind],np.unique(l_)[ind]))
    
            
            result = np.append(result,int_l1+int_l2)

    return result/(2.*np.pi)**2
