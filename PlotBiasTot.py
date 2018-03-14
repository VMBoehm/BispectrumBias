#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:12:16 2018

@author: nessa
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

thetaFWHMarcmin = 1.
noiseUkArcmin   = 1.
l_max_T         = 4000
l_max_P         = 4000
len_ang         = 800
len_l           = 5040
nums            = np.arange(5,100)#[10,20,30,35,40,45,50,55,60,70,80,85,90,93]
nums2           = np.arange(5,140)

Rpath   ='./R_files/'
Ipath   ='./outputs/integrals/'
biaspath='./biasResults/lmin2_noise1_theta10/comp_3c/'
biaspath2='./biasResults/lmin50_noise1_theta10/comp_3c/'
biaspath3='./biasResults/lmin2_noise1_theta10/comp_6c/'
#biaspath4='./biasResults/lmin2_noise6_theta14/comp_6c/'
ALpath  ='./outputs/N0files/'




if div:
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

class_file='class_cls_%s%s.pkl'%(tag,nl_)

inputpath='./outputs/ClassCls/'

#filename= Rpath+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s%s_lmax%d.pkl'%(tag,nl_,l_max_T)

#LL, Rs   = pickle.load(open(filename,'r'))
#print filename
#
#filename=Ipath+'I0I1I2kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_%s_postBorn.pkl'%bispec_tag
#params,Ls,Int0,Int2 = pickle.load(open(filename,'r'))
#print filename

bias1=[]
bias2=[]
bias3=[]
bias4=[]
Ls=[]
Ls2=[]

for ii in nums:
    L1,typea  = pickle.load(open(biaspath+'Totbias_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea2 = pickle.load(open(biaspath+'Totbias_%d_%d_%d.pkl'%(ii,len_ang/2,len_l),'r'))
    L1,typea3 = pickle.load(open(biaspath2+'Totbias_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))

    bias1+=[typea]
    bias2+=[typea2]
    bias3+=[typea3]

    Ls+=[L1]


for ii in nums2:
    L2,typea4 = pickle.load(open(biaspath3+'Totbias_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    bias4+=[typea4]
    Ls2+=[L2]

bias1=np.asarray(bias1)
bias2=np.asarray(bias2)
bias3=np.asarray(bias3)
bias4=np.asarray(bias4)
Ls =np.array(Ls)
Ls2 =np.array(Ls2)
 #minus in TypeA code

#SL        = np.interp(Ls,LL,Rs['tt']['SL'])
A_L_file  = ALpath+'%s_N0_%s_%d%d_%s%s.pkl'%(tag,lmax,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)
print A_L_file

LA, NL_KK = pickle.load(open(A_L_file,'r'))

AL        = np.interp(Ls,LA,NL_KK['tt'])
AL2       = np.interp(Ls2,LA,NL_KK['tt'])

class_params,cl_unl,cl_len = pickle.load(open(inputpath+class_file,'r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)
clphiphi2 =np.interp(Ls2,ll,clpp)



CL_bias2 = 1./(-2.*AL2)*clphiphi2
CL_bias = 1./(-2.*AL)*clphiphi

Rpath   ='./R_files/'
filename= Rpath+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s%s_lmax%d.pkl'%(tag,nl_,l_max_T)

LL, Rs   = pickle.load(open(filename,'r'))

SL       = np.interp(Ls,LL,Rs['tt']['SL'])

# bias is still a percent effect, size is very sensible to cancellation between typeA and typeC
plt.figure()
plt.loglog(Ls,abs(bias1/CL_bias),'r*')
plt.plot(Ls,abs(bias2/CL_bias),'bo',markersize=3)
plt.plot(Ls,abs(bias3/CL_bias),'g^')
plt.semilogx(Ls2,abs(bias4/CL_bias2),'c+')
plt.show()

plt.figure()
plt.plot(Ls,abs(bias2/bias1-1),'ro',label='len ang 400')
plt.semilogy(Ls,abs(bias3/bias1-1),'go',label='lmin 50')
plt.semilogy(Ls,abs(np.interp(Ls,Ls2,bias4)/bias1-1),'co',label='comp 6')
plt.axhline(y=0.1)
plt.xlim(100,3000)
plt.legend(loc='best')
plt.show()

pickle.dump([Ls,-2*AL*bias1],open('newbias1010_.pkl','w'))
