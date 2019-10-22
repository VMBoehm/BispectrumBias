# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:36:18 2017

@author: VBoehm
"""

import pickle
import pylab as pl
import numpy as np

t = lambda l: (l*(l+1.))*l**2/2./np.pi
data1= pickle.load(open('noise_levels_n10_beam40_lmax2000.pkl','r'))
#data2= pickle.load(open('noise_levels_n10_beam10_lmax5000.pkl','r'))
pl.figure()
ls = data1[0]
pl.loglog(ls, t(ls)*data1[1],label='TT')
pl.loglog(ls, t(ls)*data1[3],label='TE')
pl.loglog(ls, t(ls)*data1[4],label='TB')
pl.loglog(ls, t(ls)*data1[2],label='EE')
pl.loglog(ls, t(ls)*data1[5],label='EB')
pl.xlim(2,2000)
pl.legend(ncol=2,loc='upper left')
pl.savefig('lens_Noise_Hu_and_Okamoto_settings.png')

