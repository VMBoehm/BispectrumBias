# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:04 2017

@author: vboehm
"""

import matplotlib.pyplot as pl
import pickle
from N32biasIntegrals import I0, I1, I2
import numpy as np

#path='/afs/mpa/temp/vboehm/spectra/'
#ell_file='ell_linlog_full_1e-4_1_8000_163_cut1e-4.pkl'
#ang_file='ang_linlog_full_1e-4_1_8000_163_cut1e-4.pkl'
#filename='bispec_phi_linlog_full_1e-4_linPSPlanck2013_TempLensCombined_0-16000-0-12992241.npy'
config = 'linlog_linPs_Bfit_Planck2013_TempLensCombined'#'linlog_lnPs_Bfit_Planck2013_TempLensCombined'

#ell     = pickle.load(open(path+ell_file,'r'))
#ell     = ell[0]
#angles  = pickle.load(open(path+ang_file,'r'))
#bi_phi  = np.load(path+filename)
#angmu   = angles[3]
#mean_bispectrum = np.mean(bi_phi)
#
#
#Int0 = I0(bi_phi/mean_bispectrum, ell, angmu ,163**2, 163, squeezed=False, fullsky=False)*mean_bispectrum
#
#Int1 = I1(bi_phi/mean_bispectrum, ell, angmu ,163**2, 163, squeezed=False, fullsky=False)*mean_bispectrum            
#    
#Int2 = I2(bi_phi/mean_bispectrum, ell, angmu ,163**2, 163, squeezed=False, fullsky=False)*mean_bispectrum
# 
#L    = np.unique(ell[0::3])
#
#print L, len(L)
#print Int0, len(Int0)
#print Int2, len(Int1)
params,Limber,L,Int0,Int1,Int2=pickle.load(open('I0I1I2%s.pkl'%(config),'r'))

pl.figure()

pl.plot(L,L**6*Int0,ls='-',color='b')
pl.plot(L,L**6*Int2,ls='-',color='g')
pl.ylim(-15e-4,15e-4)
pl.xlim([50,3000])
pl.xticks([50,500,1000,2000,3000])
pl.ylabel(r'$L^6$ Integrals')
pl.xlabel(r'$L$')
pl.savefig('I0I2_%s.png'%config)