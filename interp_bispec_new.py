# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:07:14 2015

@author: Vanessa Boehm
interpolates bispectrum
Version 2: interpolates in ang13= -Lvec lvec/lL
"""

from __future__  import division
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
plt.ioff()

def bispec_interp(bispec, Ls, ls, mu, plot=True):

    bin_size=len(ls)*len(mu)

    result=[]
    # for every L interpolate over l and angle go through every L and interpolate for for every L
    for ii in np.arange(len(L)):

        spec=bispec[ii*bin_size:(ii+1)*bin_size]
        spec=np.reshape(spec,(len(l),len(mu)))
        result2=[]
        for jj in range(len(l)):
            res=splrep(mu, spec[jj], k=1)
            result2+=[res]
        result+=[result2]

        if ii in [5,10,50]:
          for kk in [20,40,90]:
             plt.figure()
             x=np.linspace(min(mu),max(mu),200)
             spec_i=splev(x ,result[ii][kk], ext=0) #ext=2: error if value outside of interpolation range
             plt.plot(x,spec_i,"ro")
             plt.plot(mu,spec[kk])
             plt.show()

    return result


path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"
filename    = path+'ells/ell_ang_full_Lmin0_Lmax3000_lmin0_lmax8000_lenL100_lenl120_lenang100_1e-04.pkl'
L,l,theta   = pickle.load(open(filename, 'r'))
print min(theta), max(theta)

tag         ='kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationcomp_12c_Lmin0-Lmax2999-lmax8000-lenBi1200000_post_born_sum'
loadfile    = path+'spectra/'+"bispec_phi_%s"%tag


print 'loading bispectrum ', loadfile

bi_phi=np.load(loadfile+'.npy')

splines = bispec_interp(bi_phi, L, l, theta)

dumpfile=path+'interp/bispec_interp_%s.pkl'%tag

pickle.dump([theta,L,l,splines],open(dumpfile,'w'))
print "dumped to %s"%dumpfile
