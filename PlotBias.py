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
nums            = [10,20,30,35,40,45,50,55,60]
bispec_tag      = 'comp_6c'


Rpath   ='./R_files/'
Ipath   ='./outputs/integrals/'
biaspath4='./biasResults/lmin2_noise6_theta14/comp_1c/'
biaspath='./biasResults/lmin2_noise6_theta14/comp_3c/'
biaspath2='./biasResults/lmin2_noise6_theta14/comp_4c/'
biaspath3='./biasResults/lmin2_noise6_theta14/comp_5c/'
biaspath5='./biasResults/lmin2_noise6_theta14/comp_6c/'
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
params,L_s,Int0,Int2 = pickle.load(open(filename,'r'))
print filename

longTypeC=np.interp(L_s,LL,Rs['tt']['para'])*Int0+np.interp(L_s,LL,Rs['tt']['perp'])*Int2
LCC=L_s
TypeC=[]
TypeC2=[]
TypeC3=[]
TypeC4=[]
TypeA=[]
TypeA2=[]
TypeA3=[]
TypeA4=[]

#print Ls[80], Ls[90], Ls[85], Ls[93]


Ls=[]
for ii in nums:

    L1,typea  = pickle.load(open(biaspath+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea2 = pickle.load(open(biaspath2+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea3 = pickle.load(open(biaspath+'A1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea4 = pickle.load(open(biaspath3+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typec  = pickle.load(open(biaspath+'C1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    L1,typec2 = pickle.load(open(biaspath+'C1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    L1,typec3 = pickle.load(open(biaspath4+'C1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    TypeA+=[typea]
    TypeA2+=[typea2]
    TypeA3+=[typea3]
    TypeA4+=[typea4]
    TypeC2+=[typec2]
    TypeC3+=[typec]
    TypeC4+=[typec3]
    Rs_para = np.interp(L1,LL,Rs['tt']['para'])
    Rs_perp = np.interp(L1,LL,Rs['tt']['perp'])
    TypeC+=[np.interp(L1,L_s,longTypeC)]
    Ls+=[L1]

L2s=Ls[:]


TypeA3l=TypeA3[:]
#TypeC4[:]
TypeA2l=TypeA2[:]
TypeCl=TypeC[:]

for ii in [80,85,90]:
    L1,typea3 = pickle.load(open(biaspath+'A1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    L1,typea2 = pickle.load(open(biaspath2+'A1_%d_%d_%d.pkl'%(ii,len_ang,len_l),'r'))
    #L1,typec3 = pickle.load(open(biaspath4+'C1_%d_%d_%d_longl.pkl'%(ii,len_ang,len_l),'r'))
    TypeCl+=[np.interp(L1,L_s,longTypeC)]
    TypeA3l+=[typea3]
    TypeA2l+=[typea2]

    L2s=np.append(L2s,L1)

L3s=[]
TypeC4l=[]
for ii in np.arange(82):
    L1,typec3 = pickle.load(open(biaspath5+'C1_%d_%d_%d.pkl'%(ii,1000,len_l),'r'))
    TypeC4l+=[typec3]
    L3s+=[L1]

Ls=Ls[:]
TypeC=np.asarray(TypeC)
TypeCl=np.asarray(TypeCl)
TypeA=np.asarray(TypeA)
TypeA2=np.asarray(TypeA2)
TypeA2l=np.asarray(TypeA2l)
TypeA3=np.asarray(TypeA3)
TypeA3l=np.asarray(TypeA3l)
TypeA4=np.asarray(TypeA4)
TypeC2=np.asarray(TypeC2)
TypeC3=np.asarray(TypeC3)
TypeC4=np.asarray(TypeC4)
TypeC4l=np.asarray(TypeC4l)
bias_sum = TypeC-TypeA2 #minus in TypeA code

SL        = np.interp(Ls,LL,Rs['tt']['SL'])
A_L_file  = ALpath+'%s_N0_%s_%d%d_%s%s.pkl'%(tag,lmax,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)
print A_L_file

LA, NL_KK = pickle.load(open(A_L_file,'r'))

AL        = np.interp(Ls,LA,NL_KK['tt'])
ALl       = np.interp(L2s,LA,NL_KK['tt'])

class_params,cl_unl,cl_len = pickle.load(open(inputpath+class_file,'r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)

clphiphil =np.interp(L2s,ll,clpp)


tot_bias  = -4.*AL**2*SL*bias_sum

CL_bias = 1./(-2.*AL)*clphiphi
CL_biasl = 1./(-2.*ALl)*clphiphil


path='/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/biasResults/OldTypeC/'

LC=[]
COl=[]
LC2=[]
COl2=[]
for ii in [84,104,116,130,140]:
  LL, C= pickle.load(open(path+'TypeC_res%d_1200_4352_simps.pkl'%ii,'r'))
  LC+=[LL]
  COl+=[C]
  try:
    LL, C= pickle.load(open(path+'TypeC_res%d_1200_4352_lin.pkl'%ii,'r'))
    LC2+=[LL]
    COl2+=[C]
  except:
    pass
COl=np.array(COl)
COl2=np.array(COl2)

plt.figure()
plt.semilogx(L2s,abs(TypeCl),'b*')
plt.plot(LC,abs(COl),'g*')
plt.plot(LC2,abs(COl2),'c*')
plt.plot(LCC,abs(longTypeC))
plt.plot(L3s,abs(TypeC4l),'r*')
plt.show()

# bias is still a percent effect, size is very sensible to cancellation between typeA and typeC
plt.figure()
plt.semilogy(Ls,abs(bias_sum/CL_bias))
plt.show()

config="Bfit_postBorn_kcut"
Ls_A,N32TypeA = pickle.load(open('../CosmoCodes/results/SimComparison/TypeA_%s'%config,'r'))
TypeAold=1./(-2.*ALl)*np.interp(L2s,Ls_A,N32TypeA)
#difference between diferent typeA calculations is mostly only a percent effect compared to the signal
plt.figure()
plt.semilogy(Ls,abs((TypeA-TypeA2)/CL_bias),'ro',label='less ang range') #closer
plt.plot(Ls,abs((TypeA-TypeA3)/CL_bias),'go',label='long l') #less close
plt.semilogx(Ls,abs((TypeA-TypeA4)/CL_bias),'co',label='more ang range') #less close
plt.plot(L2s,abs((TypeA2l-TypeA3l)/CL_biasl),'bo',label='comp 4-long l') #less close
plt.plot(L2s,abs((TypeA2l-TypeAold)/CL_biasl),'b+',label='old')
plt.legend(loc='best')
plt.show()


#difference between diferent typeC calculations is mostly only a percent effect compared to the signal
# short l high L, comp_3 is best
# long l, low L, comp 5 is best
plt.figure()
plt.semilogy(Ls,abs((TypeC+TypeC2)/TypeC),'g+', label='long l') #closer at small L
plt.semilogx(Ls,abs((TypeC+TypeC3)/TypeC),'r+', label='short l')
plt.plot(L2s,abs((TypeCl+TypeC4l)/TypeCl),'c+', label='comp 1c')
plt.legend(loc='best')
#plt.savefig('./TestPlots/TypeC_%s.png'%bispec_tag)
plt.show()

#however, their difference is important. Is it a few ercent effect or a sub percent effect?
# how do we know which is the correct TypeA or best combination of typeA and type C?
plt.figure()

plt.plot(Ls,abs((TypeA2-TypeC)/CL_bias),'bo',label='comp 4c')
plt.plot(Ls,abs((TypeA3-TypeC)/CL_bias),'co',label='long l')
plt.semilogx(Ls,abs((TypeA-TypeC)/CL_bias),'ro',label='comp 3c')
plt.semilogx(Ls,abs((TypeA4-TypeC)/CL_bias),'go',label='comp 5c')
plt.semilogx(L2s,abs((TypeAold-TypeCl)/CL_biasl),'g^',label='old')
plt.semilogx(L2s,abs((TypeA2l-TypeCl)/CL_biasl),'c^',label='A2l')
plt.semilogx(L2s,abs((TypeA2l-TypeCl)/CL_biasl),'c^',label='A2l')
plt.ylim(0.0,0.1)
#plt.semilogy(Ls,abs((TypeA2-TypeC3)/CL_bias),'b*')
#plt.semilogy(Ls,abs((TypeA3-TypeC3)/CL_bias),'c*')
#plt.semilogy(Ls,abs((TypeA-TypeC3)/CL_bias),'r*')
#plt.semilogy(Ls,abs((TypeA4-TypeC3)/CL_bias),'g*')
#
#plt.semilogy(Ls,abs((TypeA2-TypeC2)/CL_bias),'b+')
#plt.semilogy(Ls,abs((TypeA3-TypeC2)/CL_bias),'c+')
#plt.semilogy(Ls,abs((TypeA-TypeC2)/CL_bias),'r+')
#plt.semilogy(Ls,abs((TypeA4-TypeC2)/CL_bias),'g+')
plt.legend(loc='best')
plt.show()


# comp3_c and comp4_c are similar and comp5_c and comp1_c are similar at high l
plt.figure()

plt.semilogy(Ls,abs((TypeA2-TypeC)),'b*',label='comp 4c')
plt.semilogy(Ls,abs((TypeA3-TypeC)),'c*',label='long l')
plt.semilogy(Ls,abs((TypeA-TypeC)),'r*',label='comp 3c')
plt.semilogy(Ls,abs((TypeA4-TypeC)),'g*',label='comp 5c')

plt.semilogy(Ls,abs((TypeA2-TypeC3)),'bo')
plt.semilogy(Ls,abs((TypeA3-TypeC3)),'co')
plt.semilogy(Ls,abs((TypeA-TypeC3)),'ro')
plt.semilogy(Ls,abs((TypeA4-TypeC3)),'go')

plt.semilogy(Ls,abs((TypeA2-TypeC2)),'b+')
plt.semilogy(Ls,abs((TypeA3-TypeC2)),'c+')
plt.semilogy(Ls,abs((TypeA-TypeC2)),'r+')
plt.semilogy(Ls,abs((TypeA4-TypeC2)),'g+')

plt.semilogx(L2s,abs((TypeAold-TypeCl)/CL_biasl),'g^',label='old')
plt.semilogx(L2s,abs((TypeA2l-TypeCl)/CL_biasl),'c^',label='A2l')
plt.semilogx(L2s,abs((TypeA2l-TypeCl)/CL_biasl),'c^',label='A2l')
plt.legend()
#plt.ylim(1e-3,2e-1)
plt.savefig('./TestPlots/%s.png'%bispec_tag)
plt.show()

#--> run TypeC no angle cut off, longls -> need angle cut off!
#--> run TypeC no angle cut off short ls (TypeC2)
#--> comp3c, long ls, high Ls
#--> comp4c, all Ls till 93

#-> better sampling test comp_6c...
#-> test direct cancelation in code, comp3_c, short ls, with plots!



#
##config="Bfit"
##LsTypeC, N32TypeC = pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeC_%s'%config))
##
##LsTypeA,N32TypeA=pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CosmoCodes/results/SimComparison/TypeA_%s'%config, 'r'))
#
#plt.figure()
#plt.semilogy(Ls,abs(TypeA), 'ro',label='TypeA')
#plt.semilogy(Ls,abs(TypeA2), 'g+',label='TypeA')
#plt.semilogy(Ls,abs(TypeA3), 'c*',label='TypeA long l')
#plt.semilogy(Ls,abs(TypeC), 'bo',markersize=3,label='TypeC')
#plt.semilogy(Ls,abs(TypeC2), 'b*',label='TypeC long l')
##plt.semilogy(Ls,clphiphi,'bo')
##plt.ylim(1e-21,1e-8)
##plt.xlim(20,100)
#plt.legend()
#plt.show()
#print (TypeA-TypeC)/TypeA
#
#
#
#plt.figure()
#plt.semilogy(Ls,-tot_bias, 'ro')
#plt.semilogy(Ls,tot_bias,'bo')
#plt.semilogy(Ls,clphiphi)
##plt.ylim(1e-21,1e-8)
#
#plt.show()
#
#
#
##plt.figure()
##plt.plot(LL,LL.astype(float)**(-4)*2*Rs['tt']['SL'],'ro')
##plt.plot(LA,LA.astype(float)**(-4)*1./NL_KK['tt'],'b+')
##plt.show()
##
###bigger difference comes from variable redeclaration, not lensed Cls
##plt.figure()
##plt.semilogx(LA,(Rs['tt']['SL']*2-1./NL_KK['tt'])*NL_KK['tt'],'--')
##plt.plot(LA,np.zeros(len(LA)))
##plt.xlim(100,2000)
##plt.ylim(-0.4,0.1)
##plt.show()