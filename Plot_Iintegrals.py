# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:04 2017

@author: vboehm
"""

import matplotlib.pyplot as pl
import pickle
import numpy as np

config = 'kkg_g_bin0linlog_halfang_lnPs_Bfit_Planck2015_TTlowPlensing'
params,Limber,L1,Int0,Int2,bi_cum=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
print len(L1), len(Int0)
config = 'kkg_g_bin0linlog_halfang_lnPs_Bfit_Planck2015_TTlowPlensing'
path='./cross_integrals/'
config='kkg_linlog_halfangbin_0_dndz_LSST_i27_SN5_3y_lnPs_Bfit_Planck2015_TTlowPlensing'
params,Limber,L2,Int01,Int21,R,bi_cum=pickle.load(open(path+'I0I1I2%s.pkl'%(config),'r'))
config='kkg_linlog_fullangbin_0_dndz_LSST_i27_SN5_3y_lnPs_Bfit_Planck2015_TTlowPlensingtest8'
params,Limber,L,Int02,Int22,R,bi_cum=pickle.load(open(path+'I0I1I2%s.pkl'%(config),'r'))


pl.figure()
pl.plot(L1,L1**4*Int0*2,ls='-',color='m',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$')
pl.plot(L1,L1**4*Int2*2,ls='-',color='m',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel$')
pl.plot(L2,L2**4*Int01*2,ls='-',color='b',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$')
pl.plot(L2,L2**4*Int21*2,ls='-',color='g',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel$')
pl.plot(L,L**4*Int02,ls='--',color='k',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$')
pl.plot(L,L**4*Int22,ls='--',color='k',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel$')
pl.legend(loc='best',ncol=2,frameon=False)
pl.ylim(-10e-3,10e-3)
pl.xlim([50,3000])
pl.xticks([50,500,1000,2000,3000])
pl.ylabel(r'$\beta$ Integrals')
pl.xlabel(r'$L$')
pl.savefig('I0I2_%s_pB.pdf'%config)
pl.show()
