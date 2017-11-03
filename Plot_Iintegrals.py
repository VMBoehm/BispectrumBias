# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:04 2017

@author: vboehm
"""

import matplotlib.pyplot as pl
import pickle
from N32biasIntegrals import I0, I1, I2
import numpy as np

config = 'cross_g_linlog_newang_lnPs_Bfit_Planck2013_TempLensCombined_postBorn'
params,Limber,L,Int0,Int1,Int2=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
config = 'cross_g_linlog_newang_lnPs_Bfit_Planck2013_TempLensCombined'
params,Limber,L,Int01,Int11,Int21=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
config = 'linlog_lnPs_Bfit_Planck2013_TempLensCombined'
params,Limber,L,Int02,Int12,Int22=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
pl.figure()


pl.plot(L,L**4*Int01,ls='-',color='b',label=r'$L^4$ I0 N32 cross')
pl.plot(L,L**4*Int21,ls='-',color='g',label=r'$L^4$ I2 N32 cross')
pl.plot(L,L**6*Int02,ls='--',color='b',label=r'$L^6$ I0 N32')
pl.plot(L,L**6*Int22,ls='--',color='g',label=r'$L^6$ I2 N32')
pl.plot(L,L**4*Int0,ls='--',color='k',label=r'$L^4$ I0 N32 cross pB')
pl.plot(L,L**4*Int2,ls='--',color='m',label=r'$L^4$ I2 N32 cross pB')
pl.legend(loc='best',ncol=2,frameon=False)
pl.ylim(-2e-3,10e-3)
pl.xlim([50,3000])
pl.xticks([50,500,1000,2000,3000])
pl.ylabel(r'Bias Integrals')
pl.xlabel(r'$L$')
pl.savefig('I0I2_%s_pB.png'%config)


filename ='cross_spectrum_cross_g_linlog_newang_lnPs_Bfit_Planck2013_TempLensCombined'
ell, cross_spec = pickle.load(open(filename+'.pkl','r'))

pl.figure()
pl.loglog(ell,ell**2*cross_spec)
pl.ylim([1e-9,1e-6])
pl.xlim([2,2000])
pl.show()