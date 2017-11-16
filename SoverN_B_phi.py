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
 

def SN_integral(bispec, clpp, N0i, Ls, ls, Lls, ang, bin_size, sample1d, min_index, max_index, f_sky):
    # for every L interpolate over l and angle and integrate over intrepolated 2d function

    L_integrand=[]
    lmin=Ls[min_index]
    lmax=Ls[max_index]
    print "integrate L from ", lmin, "to ", lmax
    for i in np.arange(min_index,max_index):
        
        L_  = Ls[i]
        
        spec= bispec[i*bin_size:(i+1)*bin_size]
        Ll_ = Lls[i*bin_size:(i+1)*bin_size]
        integrand=[]
        
        for j in np.arange(i,max_index): #start at l=L
            l_const     = ls[j]
            
            spec_int    = spec[j*sample1d:(j+1)*sample1d]
            
            Lli         = Ll_[j*sample1d:(j+1)*sample1d]

            spec_int    = spec_int[np.where((Lli<lmax)*(Lli>l_const))] #restrict 
            ang_        = ang[np.where((Lli<lmax)*(Lli>l_const))]
            Ll          = Lli[np.where((Lli<lmax)*(Lli>l_const))]
            fac         = np.ones(len(Ll))
            
            if j==i:
                fac*=2.
                fac[np.where(np.isclose(ang_,np.pi/3.))]=6.

            integrand  += [simps(spec_int**2/(fac*(splev(L_,clpp,ext=2)+N0i(L_))*(splev(l_const,clpp,ext=2)+N0i(l_const))*(splev(Ll,clpp,ext=2)+N0i(Ll))),ang_)]

        L_integrand += [simps(integrand*ls[i:max_index],ls[i:max_index])]
        
    res = simps(L_integrand*Ls[min_index:max_index],Ls[min_index:max_index])
        
    return lmin, lmax, res
    

if __name__ == "__main__":  				
    ell_min         = 1
    ell_max         = 8000
    sample1d        = 163
    bin_size        = sample1d**2
    ell_type        = "linlog_full"
    tag             = "Planck2013_nl"
    noiseUkArcmin   = 1.
    thetaFWHMarcmin = 1.
    lcut            = 3000 
    fsky            = 0.5
         
    theta         = np.linspace(0.01,2*np.pi-0.01, sample1d)
    side1a        = np.linspace(ell_min+1,50,48,endpoint=False)
    side1b        = np.exp(np.linspace(np.log(50),np.log(ell_max),sample1d-48))
    side1         = np.append(side1a,side1b)
    
    biphi         = np.load("../results/bispec_phi_linlog_full_nlPS_Planck2013_0-16000-0-12992241_1e-2.npy")
    
    ell_file      = "../downloaded/Ll_%s_%d_%d_%d_cut1e-2.pkl"%(ell_type,ell_min,ell_max,sample1d)
    Ll            = pickle.load(open(ell_file,'r'))
    Ll            = np.array(Ll[0])
    print len(Ll)
    
    Parameter,cl_unl,cl_len=pickle.load(open('../class_outputs/class_cls_%s.pkl'%tag,'r'))
    cl_phiphi     = cl_len['pp'][2:8001]
    ells          = cl_len['ell'][2:8001]
    
    clpp_i        = splrep(ells, cl_phiphi, k=1)
    
    fields         = ['EB']#,'EE','EB']
    
    plt.figure()
    for field in fields:
        AI            = pickle.load(open('../results/lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_'+str(int(lcut))+'_%s.pkl'%tag))
        L_s           = AI[0]
        #print len(L_s), min(L_s), max(L_s)
        AL            = AI[1]
        N0            = interp1d(L_s[1:8001],AL[field][1:8001]/2.5/1.5)
        
        index_max     = 140
    
        max_L         = []
        min_L         = []
        SN            = []
        
        for index_min in [0,20,30,40,50,60,70,80,90,100]:
            minL_, maxL_, SN_ = SN_integral(biphi, clpp_i, N0, side1, side1, Ll, theta, bin_size, sample1d, index_min, index_max, fsky)
            max_L     +=[maxL_]
            min_L     +=[minL_]
            SN        +=[SN_*fsky/(2*np.pi**2)]
    #        
        print min_L, max_L, np.sqrt(SN)
    
    
        plt.plot(min_L, np.sqrt(SN),marker="o",label=field+","+field)
    plt.legend()
    plt.xlabel(r'$L_{min}$')
    plt.ylabel("S/N")
    plt.savefig("../plots/S_over_N_lmax%d_thetaFWHMarcmin%.1f_noiseUkArcmin%.1f_fsky%.1f.pdf"%(maxL_,thetaFWHMarcmin,noiseUkArcmin,fsky), bbox_inches="tight")
    plt.show()
#    
    
    #x,L_,l_,splines = pickle.load(open('../results/bispec_interp_linlog_full_nlPS_Planck2015.pkl','r'))
#    Ls, result    = betas(biphi, side1, side1, theta, bin_size, sample1d, fullsky=False)
#    print theta[1],theta[-2]
#    theta_i       = np.linspace(theta[0],theta[-1], 200)
#    #print theta[1:-1]
#    long_side     = np.append(np.arange(min(side1),1000),np.linspace(1000,4300,500))
#    
#    Ls_i, result_i= betas_interp(biphi, side1 , side1, long_side, theta_i, theta, bin_size, sample1d, fullsky=False)
#    
#    pickle.dump([Ls_i,result_i['para'],result_i['perp']],open('../results/beta_interpolated_lmax4000.pkl','w'))
    