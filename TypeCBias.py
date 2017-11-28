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
    
beta_path = '/home/traveller/Documents/Projekte/LensingBispectrum/CMB-nonlinear/I0I1I2kkg_g_bin0linlog_halfang_lnPs_Bfit_%sl_max_test10000.pkl'%tag

A_L_file='/home/traveller/Documents/Projekte/LensingBispectrum/CosmoCodes/N0files/Planck2015TempLensCombined_N0_mixedlmax_1010_nodiv.pkl'

try:
    N0 = pickle.load(open(A_L_file,'r'))
except:
    print A_L_file, ' does not exist'

try:
    bla, blub, Ls1, Iperp, Ipara = pickle.load(open(beta_path,'r'))
    Iperp*=2.
    Iperp*=2.
except:
    print beta_path, ' does not exist'
    

ll = np.arange(2,3000,dtype=float)

Iperp = np.interp(ll,Ls1,Ls1**4*Iperp)/ll**4
Ipara = np.interp(ll,Ls1,Ls1**4*Ipara)/ll**4

result={}
for field in ['tt','eb']:
    N0_    = np.interp(ll,N0['ls'],abs(N0[field]))
    Rpara  = np.interp(ll,Ls,R_integrals[field]['para'])
    Rperp  = np.interp(ll,Ls,R_integrals[field]['perp'])
    
    result[field]=-N0_*(Rperp*Iperp+Rpara*Ipara)
    
    if field=='eb':
        result['eb']*=0.5
    
    plt.figure()
    plt.plot(ll,ll**4*Iperp,label='beta perp')
    plt.plot(ll,ll**4*Ipara,label='beta para')
    plt.legend(loc='best',frameon=False)
    plt.xlim(100,3000)
    plt.show()
    
    plt.figure()
    plt.plot(ll,ll**(-2)*Rperp,label='R perp')
    plt.plot(ll,ll**(-2)*Rpara,label='R para')
    plt.legend(loc='best',frameon=False)
    plt.xlim(100,3000)
    plt.show()
    
    plt.figure()
    plt.plot(ll,ll**4*N0_,label='N0')
    plt.legend(loc='best',frameon=False)
    plt.xlim(100,3000)
    plt.show()
    
plt.figure()
plt.plot(ll,ll**4*result['tt'],label='tt')
plt.plot(ll,ll**4*result['eb'],label='eb')
plt.legend(loc='best',frameon=False)
plt.xlim(100,3000)
plt.show()

filename= 'cross_signal_noise_0_Planck2015_TTlowPlensing_nl_dndz_LSST_i27_SN5_3y.pkl'

l_,cl_xx,noise_gp = pickle.load(open(filename,'r'))

cl_xx=np.interp(ll,l_,cl_xx)
noise_gp['eb']=np.interp(ll,l_,noise_gp['eb'])
noise_gp['tt']=np.interp(ll,l_,noise_gp['tt'])

plt.figure()
plt.plot(ll,result['tt']/cl_xx,label='tt')
plt.plot(ll,result['eb']/cl_xx,label='eb')
plt.ylabel('Bias Term 2/Signal')
plt.legend(loc='best',frameon=False)
plt.ylim(-0.05,0.05)
plt.xlim(100,3000)
plt.show()

plt.figure()
plt.plot(ll,result['tt']/noise_gp['tt'],label='tt')
plt.plot(ll,result['eb']/noise_gp['eb'],label='eb')
plt.legend(loc='best',frameon=False)
plt.xlabel('L')
plt.ylabel('Bias Term 2/Noise')
plt.ylim(-0.2,0.1)
plt.xlim(100,3000)
plt.show()