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

params  = Cosmo.SimulationCosmology
tag     = params[0]['name']
field   = 'tt'

thetaFWHMarcmin = 1.4
noiseUkArcmin   = 6.
l_max_T         = 4000
l_max_P         = 4000
lmax            = 4000
len_ang         = 400
len_l           = 4096
nums            = [10,20,30]
bispec_tag      = 'comp_3c'


path='./R_files/'

filename = path+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl.pkl'%tag
LL, Rs   = pickle.load(open(filename,'r'))


path='./outputs/integrals/'

filename=path+'I0I1I2kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_%s_postBorn.pkl'%bispec_tag
params,Ls,Int0,Int2 = pickle.load(open(filename,'r'))


TypeC=[]
TypeA=[]

for ii in nums:
    L=Ls[ii]
    Rs_para = np.interp(L,LL,Rs['tt']['para'])
    Rs_perp = np.interp(L,LL,Rs['tt']['perp'])
    TypeC+=[Rs_perp*Int0[ii]+Rs_para*Int2[ii]]
    L1,typea= pickle.load(open(path+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    TypeA+=typea
    assert(L1==L)

Ls=Ls[nums]
TypeC=np.asarray(TypeC)
TypeA=np.asarray(TypeA)
bias_sum = TypeC-TypeA #minus in TypeA code

A_L_file  = './outputs/N0files/%s_N0_%s_%d%d_nodiv.pkl'%(tag,lmax,10*noiseUkArcmin,10*thetaFWHMarcmin)

LL, NL_KK = pickle.load(open(A_L_file,'r'))

AL        = np.interp(Ls,LL,NL_KK['tt'])
SL        = np.interp(Ls,LL,Rs['tt']['Ls'])

tot_bias  = -4.*AL**2*SL*bias_sum

class_params,cl_unl,cl_len = pickle.load(open('../Simulations/NewRuns/CMBLensSims/inputParams/class_cls_Jias_Simulation_new_nl.pkl','r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)

plt.figure()
plt.semilogy(Ls,-tot_bias, 'ro')
plt.semilogy(Ls,tot_bias, 'go')
plt.semilogy(Ls,clphiphi,'bo')
#plt.ylim(1e-21,1e-8)
plt.xlim(20,100)
plt.show()




