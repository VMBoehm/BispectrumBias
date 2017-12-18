# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:04 2017

@author: vboehm
"""

import matplotlib.pyplot as pl
import pickle
import numpy as np

ell_type='linlog_halfang'
i=1.
for red_bin in ['1','0','2','None']:
    
    if red_bin=='None':
        LSST = False
    else:
        LSST = True  
        dn_filename = 'dndz_LSST_i27_SN5_3y'
    conf = 'kkg_%s'%ell_type
    if LSST:
        conf+='bin_%s_%s'%(red_bin,dn_filename)
    else:
        conf+='no_binning'

    path='./cross_integrals/'
    config = '%s_lnPs_Bfit_Planck2015_TTlowPlensing_postBorn_only'%conf
    params,Limber,L1,Int0,Int2=pickle.load(open(path+'I0I1I2%s.pkl'%(config),'r'))
    config='%s_lnPs_Bfit_Planck2015_TTlowPlensing'%conf
    params,Limber,L2,Int01,Int21,R,bi_cum=pickle.load(open(path+'I0I1I2%s.pkl'%(config),'r'))
    config='%s_lnPs_Bfit_Planck2015_TTlowPlensing_postBorn'%conf
    params,Limber,L0,Int02,Int22=pickle.load(open(path+'I0I1I2%s.pkl'%(config),'r'))


    pl.figure()
    pl.plot(L1,L1**4*Int0*2,ls='-',color='r',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$ PB')
    pl.plot(L1,L1**4*Int2*2,ls='--',color='r',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel$ PB')
    pl.plot(L2,L2**4*Int01*2,ls='-',color='g',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$ NL')
    pl.plot(L2,L2**4*Int21*2,ls='--',color='g',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel $ NL')
    pl.plot(L0,L0**4*Int02*2,ls='-',color='orange',label=r'$L^4 \beta^{\mathrm{cross}}_\perp$ NL+PB')
    pl.plot(L0,L0**4*Int22*2,ls='--',color='orange',label=r'$L^4 \beta^{\mathrm{cross}}_\parallel$ NL+PB')
    
    
    pl.legend(loc='best',ncol=2,frameon=False)
    pl.ylim(-20e-3*i,7e-3)
    pl.xlim([50,3000])
    pl.xticks([50,500,1000,2000,3000])
    pl.ylabel(r'$\beta$ Integrals')
    pl.xlabel(r'$L$')
    pl.savefig('I0I2_%s_pB.pdf'%conf)
    pl.show()
    i=1.
