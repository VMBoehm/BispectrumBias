# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 03:15:39 2017

@author: VBoehm
"""
from __future__ import division
import numpy as np
import pickle
import Cosmology as Cosmo
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

params=Cosmo.Planck2015_TTlowPlensing
tag=params[0]['name']
fields = ['tt','eb']

thetaFWHMarcmin = 1. #beam FWHM
noiseUkArcmin = 1.#eval(sys.argv[1]) #Noise level in uKarcmin
l_max_T       = 3000
l_max_P       = 5000


R_path='./R_files/'

R_filename = R_path+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl'%tag
try:
    Ls,R_integrals = pickle.load(open(R_filename+'.pkl','r'))
except:
    print R_filename+'.pkl', ' does not exist'

for zbin,LSST in zip(['0','1','2','None'],[True,True,True,False]):
    if LSST:
        zbin='bin_%s'%zbin
        tag='_dndz_LSST_i27_SN5_3y'
    else:
        zbin='no_binning'
        tag=''
    beta_path ='/home/traveller/Documents/Projekte/LensingBispectrum/CMB-nonlinear/cross_integrals/I0I1I2kkg_linlog_halfangbin_0_dndz_LSST_i27_SN5_3y_lnPs_Bfit_Planck2015_TTlowPlensingextr1_postBorn.pkl'#I0I1I2kkg_linlog_halfang%s%s_lnPs_Bfit_Planck2015_TTlowPlensing.pkl'%(zbin,tag)
    
    print beta_path
    
    A_L_file='/home/traveller/Documents/Projekte/LensingBispectrum/CosmoCodes/N0files/Planck2015TempLensCombined_N0_mixedlmax_1010_nodiv.pkl'
    
    try:
        N0 = pickle.load(open(A_L_file,'r'))
    except:
        print A_L_file, ' does not exist'
    

    bla, blub, Ls1, Iperp, IPara = pickle.load(open(beta_path,'r'))
    Iperp*=2.
    Iperp*=2.

        
    
    ll = Ls1#np.arange(2,3000,dtype=float)
    
    Iperp = np.interp(ll,Ls1,Iperp)
    #~ 5% error du to smoothing
    if zbin=='0':
        #Ipara = splrep(Ls1,Ls1**4*IPara,k=1)#,k=5,s=2e-29)
        #Ipara = splev(ll,Ipara)/ll**4
        Ipara = np.interp(ll,Ls1,IPara)
#        plt.figure()
#        plt.semilogx(ll, 100*(Ipara-IPara)/IPara)
#        plt.ylim(-20,20)
#        plt.xlim(100,2000)
#        plt.show()
    else:
        #Ipara = splrep(Ls1,Ls1**4*IPara,k=1)
        #Ipara = splev(ll,Ipara)/ll**4
        Ipara = np.interp(ll,Ls1,IPara)
#        plt.figure()
#        plt.semilogx(ll, 100*(Ipara-IPara)/IPara)
#        plt.ylim(-20,20)
#        plt.xlim(100,2000)
#        plt.show()
    
    
    
    result={}
    for field in ['tt']:
        N0_    = np.interp(ll,N0['ls'],N0['ls']**4*abs(N0[field]))/ll**4
        Rpara  = np.interp(ll,Ls,Ls**(-2)*R_integrals[field]['para'])*ll**2
        Rperp  = np.interp(ll,Ls,Ls**(-2)*R_integrals[field]['perp'])*ll**2
        
        result[field]=-N0_*(Rperp*Iperp+Rpara*Ipara)
        
        if field=='eb':
            result['eb']*=0.5
        
        plt.figure()
        plt.plot(ll,ll**2*Iperp,label='beta perp')
        plt.loglog(ll,-ll**2*Ipara,label='beta para')
        plt.loglog(ll,-ll**2*Ipara,label='beta para')
        plt.loglog(Ls1,-Ls1**2*IPara,label='beta para',ls='',marker='o')
        plt.loglog(Ls1,-Ls1**2*IPara,label='beta para',ls='',marker='o')
        plt.loglog(Ls1[81],-Ls1[81]**2*IPara[81],label='beta para',ls='',marker='o')
        plt.axvline(x=330)
        plt.legend(loc='best',frameon=False)
        plt.xlim(100,3000)
        plt.show()
        print Ls1[81]
#        
        plt.figure()
        plt.plot(ll,ll**4*Rperp,label='R perp')
        plt.loglog(ll,ll**4*Rpara,label='R para')
        plt.plot(ll,-ll**4*Rperp,label='R perp')
        plt.loglog(ll,-ll**4*Rpara,label='R para')
        plt.axvline(x=329)
        plt.legend(loc='best',frameon=False)
        plt.xlim(100,3000)
        plt.show()
        
        plt.figure()
        plt.loglog(ll,ll**4*N0_,label='N0')
        plt.legend(loc='best',frameon=False)
        plt.axvline(x=330)
        plt.xlim(100,3000)
        plt.show()
        
    
    filename= 'cross_signal_noise_0_Planck2015_TTlowPlensing_nl_dndz_LSST_i27_SN5_3y.pkl'
    
    l_,cl_xx,noise_gp = pickle.load(open(filename,'r'))
    
    cl_xx=np.interp(ll,l_,cl_xx)
    #noise_gp['eb']=np.interp(ll,l_,noise_gp['eb'])
    noise_gp['tt']=np.interp(ll,l_,noise_gp['tt'])
    
    plt.figure()
    plt.semilogx(ll,result['tt']/cl_xx,label='tt')
    #plt.plot(ll,result['eb']/cl_xx,label='eb')
    plt.ylabel('Bias Term 2/Signal')
    plt.legend(loc='best',frameon=False)
    plt.ylim(-0.05,0.05)
    plt.xlim(100,3000)
    plt.show()
    
    plt.figure()
    plt.semilogx(ll,result['tt']/noise_gp['tt'],label='tt')
    #plt.plot(ll,result['eb']/noise_gp['eb'],label='eb')
    plt.legend(loc='best',frameon=False)
    plt.xlabel('L')
    plt.ylabel('Bias Term 2/Noise')
    plt.ylim(-0.11,0.05)
    plt.xlim(100,3000)
    plt.show()
    
    plt.figure()
    plt.semilogx(ll,ll**4*result['tt'],label='tt')
    #plt.plot(ll,ll**4*result['eb'],label='eb')
    #plt.axvline(x=330)
    plt.legend(loc='best',frameon=False)
    plt.xlabel('L')
    plt.ylabel('Bias Term 2')
    plt.xlim(100,3000)
    plt.ylim(-0.001,0.0015)
    plt.savefig('BiasTerm2.png')
    plt.show()

for ii in [20,30]:
    typeA=pickle.load(open('/home/traveller/Documents/Projekte/LensingBispectrum/CMB-nonlinear/TypeAscripts/cross_results/TypeA_res%d_1400_4608_kkg_linlog_halfangbin_0_dndz_LSST_i27_SN5_3y_lnPs_Bfit_Planck2015_TTlowPlensing_postBorn_sum_4bias_mu.pkl'%ii,'r'))
    print typeA
    L=Ls1[ii]
    AL=np.interp(L,N0['ls'],abs(N0['tt']))
    print L, np.interp(L,ll,result['tt']), AL*typeA[1]

    