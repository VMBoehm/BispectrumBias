# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:04 2017

@author: vboehm
"""

import matplotlib.pyplot as pl
import pickle
from N32biasIntegrals import I0, I1, I2
import numpy as np

config = 'cross_g_bin0linlog_newang_lnPs_Bfit_Planck2013_TempLensCombined'
params,Limber,L,Int0,Int1,Int2=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
config = 'cross_g_linlog_newang_lnPs_Bfit_Planck2013_TempLensCombined'
params,Limber,L,Int01,Int11,Int21=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
config = 'linlog_lnPs_Bfit_Planck2013_TempLensCombined'
params,Limber,L,Int02,Int12,Int22=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
pl.figure()


pl.plot(L,L**4*Int01,ls='-',color='b',label=r'$L^4 \beta_0$ N32 cross')
pl.plot(L,L**4*Int21,ls='-',color='g',label=r'$L^4 \beta_2$ N32 cross')
pl.plot(L,L**6*Int02,ls='--',color='b',label=r'$L^6 \beta_0$ N32')
pl.plot(L,L**6*Int22,ls='--',color='g',label=r'$L^6 \beta_2$ N32')

pl.plot(L,L**4*Int0,ls='--',color='k',label=r'$L^4 \beta_0$ N32 cross bin0')
pl.plot(L,L**4*Int2,ls=':',color='m',label=r'$L^4 \beta_2$ N32 cross bin0')
config = 'linlog_halfang_lnPs_Bfit_Namikawa_Paper'
params,Limber,L,Int03,Int13,Int23=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))
pl.plot(L,L**6*Int03*2,ls=':',color='b',label=r'$L^6 \beta_0$ N32')
pl.plot(L,L**6*Int23*2,ls=':',color='g',label=r'$L^6 \beta_2$ N32')
pl.legend(loc='best',ncol=2,frameon=True)
pl.ylim(-2e-3,10e-3)
pl.xlim([50,3000])
pl.xticks([50,500,1000,2000,3000])
pl.ylabel(r'Bias Integrals')
pl.xlabel(r'$L$')
pl.savefig('I0I2_%s_pB.pdf'%config)




filename ='cross_spectrum_Planck2013_TempLensCombined_nl_dndz_LSST_i27_SN5_3y_bin0'
ell, spec_lens, spec_gg, cross_spec  = pickle.load(open(filename+'.pkl','r'))


Parameter,cl_unl,cl_len=pickle.load(open('../class_outputs/class_cls_Planck2013_TempLensCombined_nl.pkl','r'))
cl_phiphi     = cl_len['pp'][2:8001]
ells          = cl_len['ell'][2:8001]

pl.figure()
pl.loglog(ell,1./2.*(ell*(ell+1))*cross_spec, color='b',ls='--',label=r'$C_L^{\kappa g}$, z=0-0.5')
pl.loglog(ell,spec_gg, color='b',ls='-', label=r'$C_L^{gg}$, z=0-0.5')
pl.loglog(ell,1./4.*(ell*(ell+1.))**2*spec_lens, 'k',label=r'$C_L^{\kappa \kappa}$')
filename='cross_spectrum_Planck2013_TempLensCombined_nl_dndz_LSST_i27_SN5_3y_bin1'
ell, spec_lens, spec_gg, cross_spec = pickle.load(open(filename+'.pkl','r'))
pl.loglog(ell,1./2.*(ell*(ell+1.))*cross_spec, color='g',ls='--', label=r'$C_L^{\kappa g}$, z=0.5-1.')
pl.loglog(ell,spec_gg, color='g',ls='-', label=r'$C_L^{gg}$, z=0.5-1.')
pl.loglog(ells,1./4.*(ells*(ells+1.))**2*cl_phiphi, 'r',label=r'$C_L^{\kappa \kappa}$ theory')
pl.legend(loc='lower left',ncol=2, columnspacing=0.8, frameon=True)
pl.ylim([1e-9,1e-4])
pl.xlim([2,2000])
pl.xlabel('L')
pl.savefig('Cross_Spectra_%s.pdf'%config)
pl.show()