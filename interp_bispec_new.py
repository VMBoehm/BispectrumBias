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
            Ll  = np.sqrt(L[ii]**2+l[jj]**2-2.*L[ii]*l[jj]*np.cos(mu))
            res = splrep(mu, spec[jj]/l[jj]**2/Ll**2*4., k=1) #convert kkg to ppg
            result2+=[res]
        result+=[result2]

        if ii in [5,10,50]:
          for kk in [20,40,90]:
             plt.figure()
             x=np.linspace(min(mu),max(mu),200)
             Ll  = np.sqrt(L[ii]**2+l[kk]**2-2.*L[ii]*l[kk]*np.cos(mu))
             spec_i=splev(x ,result[ii][kk], ext=0) #ext=2: error if value outside of interpolation range
             plt.plot(x,spec_i,"ro")
             plt.plot(mu,spec[kk]/l[kk]**2/Ll**2*4.)
             plt.show()

    return result


path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"
filename    = path+'ells/ell_ang_full_Lmin1_Lmax3000_lmin1_lmax8000_lenL120_lenl140_lenang120_1e-04.pkl'
L,l,theta   = pickle.load(open(filename, 'r'))
print min(theta), max(theta)

tag         = 'cross_bias_gal_LSSTbinall_full_Planck2015_Lmin1-Lmax2999-lmax8000_halofit_SC_post_born_sum'
loadfile    = path+'bispectra/'+"bispec_%s"%tag


print 'loading bispectrum ', loadfile

bi_phi=np.load(loadfile+'.npy')

splines = bispec_interp(bi_phi, L, l, theta)

dumpfile=path+'interp/bispec_interp_%s.pkl'%tag

pickle.dump([theta,L,l,splines],open(dumpfile,'w'))
print "dumped to %s"%dumpfile
