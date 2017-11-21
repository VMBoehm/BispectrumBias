# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:29:14 2016
SoverN_Bphi.py
Computes S/N of B_phi in flat sky for simple Gaussian Noise
@author: VBoehm
"""

import pylab as plt
import pickle
from scipy.integrate import simps
import numpy as np
from scipy.interpolate import splev, splrep, interp1d
import Cosmology as Cosmo
               
def SN_integral(bispec,var_len, var_gal, var_xx, Ls, ls, Lls, ang, bin_size, sample1d, min_index, max_index):
    # for every L interpolate over l and angle and integrate over intrepolated 2d function

    L_integrand=[]
    lmin=Ls[min_index]
    lmax=Ls[max_index]
    print "integrate L from ", lmin, "to ", lmax
    for i in np.arange(min_index,max_index):
        L_      = Ls[i]
        spec    = bispec[i*bin_size:(i+1)*bin_size]
        Ll_     = Lls[i*bin_size:(i+1)*bin_size]
        
        integrand = []
        
        for j in np.arange(i,max_index): #start at l=L
            l_const     = ls[j]
            spec_int    = spec[j*sample1d:(j+1)*sample1d]
            
            Lli         = Ll_[j*sample1d:(j+1)*sample1d]
            
            index       = np.where((Lli<lmax)*(Lli>l_const))

            spec_int    = spec_int[index] #restrict 
            ang_        = ang[index]
            Ll          = Lli[index]
            
            ll          = np.arange(np.floor(min(Ll)),np.ceil(max(Ll)))
#            if len(ll)<=1:
#                ll          = np.arange(np.floor(min(Ll)),np.ceil(max(Ll)))
            fac         = np.ones(len(ll))
            
            if j==i:
                fac*=2.
                fac[np.where(np.isclose(l_const,ll))]=6.
            
            L       = int(l_const)+int(L_)+ll
            L       = 0.5*L
            full_s  = (-1)**L*np.sqrt(np.exp(1)/(2.*np.pi))*(L+1)**(-0.25)
            full_s*=(L-l_const+1)**(-0.25)*(L-L_+1)**(-0.25)*(L-ll+1)**(-0.25)
            full_s*=((L-l_const+0.5)/(L-l_const+1))**(L-l_const+0.25)
            full_s*=((L-L_+0.5)/(L-L_+1))**(L-L_+0.25)
            full_s*=((L-ll+0.5)/(L-ll+1))**(L-ll+0.25)
            full_s*=np.sqrt((2*L_+1)*(2*l_const+1)*(2*ll+1)/np.pi/4.)
            full_s[np.where(((2*L))%2!=0)]=0.


            spec_ = np.interp(ll,Ll,spec_int)
            num   = (2.*full_s*spec_)**2
            denom = fac*splev(L_,var_lens,ext=0)*splev(l_const,var_lens,ext=0)*splev(ll,var_len,ext=0)

            
            integrand+=[sum(num/denom)]
#            else:
#
#                integrand+=[0]
        ll          = np.arange(np.floor(min(ls[i:max_index])),np.ceil(max(ls[i:max_index])))
        integrand   = np.interp(ll,ls[i:max_index],integrand)
        
        L_integrand += [sum(integrand)]
        
    ll  = np.arange(np.floor(min(Ls[min_index:max_index])),np.ceil(max(Ls[min_index:max_index])))
    L_integrand = np.interp(ll,Ls[min_index:max_index],L_integrand)
    res = sum(L_integrand)
    print max(ll)
        
    return lmin, lmax, res
    

if __name__ == "__main__":  				
    
    red_bin     = '0'
    params      = Cosmo.Namikawa#Planck2015_TTlowPlensing
    tag         = params[0]['name']+'_nl'  
    dn_filename = 'dndz_LSST_i27_SN5_3y'
    
    ell_min     = 1
    ell_max     = 3000
    len_L       = 160
    len_ang     = 160
    Delta_theta = 1e-2
    
    La          = np.linspace(ell_min,50,48,endpoint=False)
    Lb          = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
    side1       = np.append(La,Lb)
    
    fsky        = 0.5
    
    theta       = np.linspace(Delta_theta,np.pi, len_ang)
    
    ll,var_lens,var_gal,cl_xx = pickle.load(open('Gaussian_variances_CMB-S4_LSST_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'r'))
    
    
    Parameter,cl_unl,cl_len   = pickle.load(open('../class_outputs/class_cls_%s.pkl'%tag,'r'))
    
    b_kkg       = np.load("bispec_phi_linlog_halfang_lnPs_Bfit_Namikawa_Paper_lmin1-lmax2999-lenBi4096000.npy")     
    Ll          = np.asarray(np.load(open('Ll_file_linlog_halfang_1e+00_3000_lenL160_lenang160_1e-02.pkl','r')))

    var_gal     = splrep(ll,var_gal)
    var_lens    = splrep(ll,var_lens)
    var_xx      = splrep(ll,cl_xx)
    
    min_L       = []
    SN          = []
    index_max   = 159
    
#    for index_min in np.arange(2,159,10):
#        print index_min, index_max
#        minL_, maxL_, SN_ = SN_integral(b_kkg, var_lens, var_gal, var_xx, side1, side1, Ll, theta, len_L**2, len_L, index_min, index_max, fsky)
#        min_L     +=[minL_]
#        SN        +=[SN_*fsky/(2*np.pi**2)]
##        
#    
#    
#        plt.semilogx(min_L, np.asarray(min_L)*SN/max(SN) ,marker="o")
#    plt.legend()
#    plt.xlabel(r'$L_{min}$')
#    plt.ylabel("S/N")
#    plt.savefig("S_over_N_B_phi_lmax%d_thetaFWHMarcmin10_noiseUkArcmin10_fsky%.1f.pdf"%(maxL_,fsky), bbox_inches="tight")
#    plt.show()
    
    max_L       = []
    SN          = []
    index_min   = 0
    
    for index_max in np.arange(80,159,5):
        print index_min, index_max
        minL_, maxL_, SN_ = SN_integral(b_kkg, var_lens, var_gal, var_xx, side1, side1, Ll, theta, len_L**2, len_L, index_min, index_max)
        max_L     +=[side1[index_max-1]]
        print max_L
        SN        +=[SN_*fsky]
#        
    
    
        plt.plot(max_L, np.sqrt(SN) ,marker="o")
    plt.xlim(0,2000)
    plt.legend()
    plt.xlabel(r'$L_{max}$')
    plt.ylabel("S/N")
    plt.savefig("S_over_N_B_phi_lmin%d_thetaFWHMarcmin10_noiseUkArcmin10_fsky%.1f.pdf"%(minL_,fsky), bbox_inches="tight")
    plt.show()
