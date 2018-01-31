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

        for jj in range(len(l)):
            res=splrep(mu, spec[jj], k=1)
            result+=[res]
            if plot:
                if ii in [5,10,50] and jj in [5,20,50]:
                    plt.figure()
                    x=np.linspace(0,2*np.pi,200)
                    spec_i=splev(x ,result[-1], ext=0) #ext=2: error if value outside of interpolation range
                    plt.plot(x,spec_i,"ro")
                    plt.plot(mu,spec[jj])
                    plt.show()

    return result


path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"
filename    = path+'ell_ang_full_Lmin1_Lmax3000_lmin1_lmax8000_lenL100_lenl120_lenang100_1e-04.pkl'
L,l,theta   = pickle.load(open(filename, 'r'))

tag         = 'kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_comp_1_Lmin1-Lmax2999-lmax8000-lenBi1200000'
loadfile    = path+'spectra/'+"bispec_phi_%s"%tag


print 'loading bispectrum ', loadfile

bi_phi=np.load(loadfile+'.npy')

splines = bispec_interp(bi_phi, L, l, theta)

dumpfile=path+'bispec_interp_%s.pkl'%tag

pickle.dump([theta,L,l,splines],open(dumpfile,'w'))
print "dumped to %s"%dumpfile
