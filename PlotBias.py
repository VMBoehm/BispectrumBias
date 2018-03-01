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

thetaFWHMarcmin = 1.4
noiseUkArcmin   = 6.
l_max_T         = 4000
l_max_P         = 4000
len_ang         = 400
len_l           = 4560
nums            = [30,35,40,45,50,55,60,65]
bispec_tag      = 'comp_3c'


Rpath   ='./R_files/'
Ipath   ='./outputs/integrals/'
biaspath='./biasResults/lmin2_noise6_theta14/comp_3c/'
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

filename= Rpath+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s%s_lmax%d.pkl'%(tag,nl_,l_max_T)

LL, Rs   = pickle.load(open(filename,'r'))
print filename

filename=Ipath+'I0I1I2kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_%s_postBorn.pkl'%bispec_tag
params,Ls,Int0,Int2 = pickle.load(open(filename,'r'))
print filename

TypeC=[]
TypeA=[]
TypeA2=[]

for ii in nums:
    print ii
    L=Ls[ii]
    print L
    Rs_para = np.interp(L,LL,Rs['tt']['para'])
    Rs_perp = np.interp(L,LL,Rs['tt']['perp'])
    TypeC+=[Rs_perp*Int0[ii]+Rs_para*Int2[ii]]
    L1,typea= pickle.load(open(biaspath+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea2= pickle.load(open(biaspath+'A1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    TypeA+=[typea]
    TypeA2+=[typea2]
    assert(L1==L)

Ls=Ls[nums]
TypeC=np.asarray(TypeC)
TypeA=np.asarray(TypeA)
TypeA2=np.asarray(TypeA2)
bias_sum = TypeC-TypeA #minus in TypeA code

SL        = np.interp(Ls,LL,Rs['tt']['SL'])
A_L_file  = ALpath+'%s_N0_%s_%d%d_%s%s.pkl'%(tag,lmax,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)
print A_L_file

LA, NL_KK = pickle.load(open(A_L_file,'r'))

AL        = np.interp(Ls,LA,NL_KK['tt'])


tot_bias  = -4.*AL**2*SL*bias_sum




plt.figure()
plt.plot(Ls,(TypeA-TypeA2)/TypeA,'ro')
plt.show()

plt.figure()
plt.semilogy(Ls,abs((TypeA-TypeC)/TypeC),'ro')
plt.semilogy(Ls,abs((TypeA2-TypeC)/TypeC),'bo')
plt.show()

class_params,cl_unl,cl_len = pickle.load(open(inputpath+class_file,'r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)

#config="Bfit"
#LsTypeC, N32TypeC = pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeC_%s'%config))
#
#LsTypeA,N32TypeA=pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeA_%s'%config, 'r'))

plt.figure()
plt.semilogy(Ls,abs(TypeA), 'ro',label='TypeA')
plt.semilogy(Ls,abs(TypeA2), 'g+',label='TypeA')
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



#plt.figure()
#plt.plot(LL,LL.astype(float)**(-4)*2*Rs['tt']['SL'],'ro')
#plt.plot(LA,LA.astype(float)**(-4)*1./NL_KK['tt'],'b+')
#plt.show()
#
##bigger difference comes from variable redeclaration, not lensed Cls
#plt.figure()
#plt.semilogx(LA,(Rs['tt']['SL']*2-1./NL_KK['tt'])*NL_KK['tt'],'--')
#plt.plot(LA,np.zeros(len(LA)))
#plt.xlim(100,2000)
#plt.ylim(-0.4,0.1)
#plt.show()