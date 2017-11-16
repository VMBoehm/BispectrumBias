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
               
def SN_integral(bispec,var_len, var_gal, var_xx, Ls, ls, Lls, ang, bin_size, sample1d, min_index, max_index, f_sky):
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
            fac         = np.ones(len(Ll))
            
            if j==i:
                fac*=2.
                fac[np.where(np.isclose(ang_,np.pi/3.))]=6.
                
            num   = spec_int**2
            denom = fac*var_lens(L_)*var_lens(l_const)*var_len(Ll)
            integrand+=[simps(num/denom,ang_)]

                
        L_integrand += [simps(integrand*ls[i:max_index],ls[i:max_index])]
        
    res = simps(L_integrand*Ls[min_index:max_index],Ls[min_index:max_index])
        
    return lmin, lmax, res
    

if __name__ == "__main__":  				
    
    red_bin     = '0'
    params      = Cosmo.Planck2015_TTlowPlensing
    tag         = params[0]['name']+'_nl'  
    dn_filename = 'dndz_LSST_i27_SN5_3y'
    
    ell_min     = 2
    ell_max     = 3000
    len_L       = 163
    len_ang     = 163
    Delta_theta = 1e-2
    
    La          = np.linspace(ell_min,50,48,endpoint=False)
    Lb          = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
    side1       = np.append(La,Lb)
    
    fsky        = 0.5
    
    theta       = np.linspace(Delta_theta,2*np.pi-Delta_theta, len_ang)
    
    ll,var_lens,var_gal,cl_xx = pickle.load(open('Gaussian_variances_CMB-S4_LSST_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'r'))
    
    Parameter,cl_unl,cl_len   = pickle.load(open('../class_outputs/class_cls_%s.pkl'%tag,'r'))
    
    b_kkg       = np.load("bispec_phi_linlog_newang_lnPs_Bfit_Planck2015_TTlowPlensing_lmin2-lmax2999-lenBi4330747.npy")
    
    Ll          = np.asarray(np.load(open('Ll_file_linlog_newang_2e+00_3000_lenL163_lenang163_1e-02.pkl','r')))
    
    var_gal     = splrep(ll,var_gal)
    var_lens    = interp1d(ll,var_lens)
    var_xx      = splrep(ll,cl_xx)
    
    min_L       = []
    SN          = []
    index_max   = 162
    
    for index_min in np.arange(10,140,20):
        print index_min, index_max
        minL_, maxL_, SN_ = SN_integral(b_kkg, var_lens, var_gal, var_xx, side1, side1, Ll, theta, len_L**2, len_L, index_min, index_max, fsky)
        min_L     +=[minL_]
        SN        +=[SN_*fsky/(2*np.pi**2)]
#        
    
    
        plt.plot(min_L, np.sqrt(SN) ,marker="o")
    plt.legend()
    plt.xlabel(r'$L_{max}$')
    plt.ylabel("S/N")
    plt.savefig("S_over_N_B_phi_lmax%d_thetaFWHMarcmin10_noiseUkArcmin10_fsky%.1f.pdf"%(maxL_,fsky), bbox_inches="tight")
    plt.show()
