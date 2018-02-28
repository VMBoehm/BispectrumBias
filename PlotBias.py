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
nl      = True
div     = False

thetaFWHMarcmin = 7
noiseUkArcmin   = 30.
l_max_T         = 2000
l_max_P         = 2000
len_ang         = 400
len_l           = 4560
nums            = [10,20,30,35,40,45,50,55,60,65]
bispec_tag      = 'comp_3c'


Rpath   ='./R_files/'
Ipath   ='./outputs/integrals/'
biaspath='./biasResults/lmin2_noise6_theta14/comp_3c/'
ALpath  ='./outputs/N0files/'


if div:
    print 'Dividing EB by factor 2.5!'
    NL_KK['EB']*=1./2.5
    no_div='div25'
else:
    no_div='nodiv'

if l_max_T!=l_max_P:
    lmax='mixedlmax'
else:
    lmax=str(l_max_T)


if nl:
  nl_='_nl'
else:
  nl_=''

filename= Rpath+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s%s_lmax%d.pkl'%(tag,nl_,l_max_T)

LL, Rs   = pickle.load(open(filename,'r'))
print filename

filename=Ipath+'I0I1I2kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_%s_postBorn.pkl'%bispec_tag
params,Ls,Int0,Int2 = pickle.load(open(filename,'r'))
print filename

TypeC=[]
TypeA=[]

for ii in nums:
    print ii
    L=Ls[ii]
    print L
    Rs_para = np.interp(L,LL,Rs['tt']['para'])
    Rs_perp = np.interp(L,LL,Rs['tt']['perp'])
    TypeC+=[Rs_perp*Int0[ii]+Rs_para*Int2[ii]]
    L1,typea= pickle.load(open(biaspath+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    TypeA+=[typea]
    assert(L1==L)

Ls=Ls[nums]
TypeC=np.asarray(TypeC)
TypeA=np.asarray(TypeA)
bias_sum = TypeC-TypeA #minus in TypeA code

SL        = np.interp(Ls,LL,Rs['tt']['SL'])
A_L_file  = ALpath+'%s_N0_%s_%d%d_%s%s.pkl'%(tag,lmax,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)
print A_L_file

LA, NL_KK = pickle.load(open(A_L_file,'r'))

AL        = np.interp(Ls,LA,NL_KK['tt'])


tot_bias  = -4.*AL**2*SL*bias_sum

class_params,cl_unl,cl_len = pickle.load(open('../Simulations/NewRuns/CMBLensSims/inputParams/class_cls_Jias_Simulation_new_nl.pkl','r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)

config="Bfit"
LsTypeC, N32TypeC = pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeC_%s'%config))

LsTypeA,N32TypeA=pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeA_%s'%config, 'r'))

plt.figure()
plt.semilogy(Ls,abs(TypeA), 'ro',label='TypeA')
plt.semilogy(Ls,abs(TypeC), 'bo',markersize=3,label='TypeC')
#plt.semilogy(Ls,clphiphi,'bo')
#plt.ylim(1e-21,1e-8)
#plt.xlim(20,100)
plt.legend()
plt.show()
print (TypeA-TypeC)/TypeA



plt.figure()
plt.semilogy(Ls,-tot_bias, 'ro')
plt.semilogy(Ls,tot_bias,'bo')
plt.semilogy(Ls,clphiphi)
#plt.ylim(1e-21,1e-8)

plt.show()

plt.figure()
plt.plot(LL,LL**(-4)*2*Rs['tt']['SL'])
plt.plot(LA,LA.astype(float)**(-4)*1./NL_KK['tt'],ls='--')
plt.xlim(100,4000)
plt.ylim(5e3,6e5)
plt.show()
