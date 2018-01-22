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
import Cosmology as C
plt.ioff()

def bispec_interp(bispec, L,l, ang, len_l, len_ang, len_L,plot=True):
    """interpolates the bispectrum computed in Spectra.py
    * bispec: Bispectrum(ell[0::3], ell[1::3],ell[2::3])
    * ell: arguments to bispec
    * ang13: angle between L and l
    * bin size: len(ell)
    * sample1d: length of unique L array
    """


    result=[]
    bin_size=len_l*len_ang


    # for every L interpolate over l and angle go through every L and interpolate for for every L
    for i in np.arange(len_L):
        i  = np.int(i)

        spec=bispec[i*bin_size:(i+1)*bin_size]
        spec=np.reshape(spec,(len_ang,len_l))

        result2=[]
        #for every fixed l interpolate in angle
        for j in range(len_ang):
            result2+=[splrep(l, spec[j], k=1)]
            if plot:
                if i in [0,int(len(L)/3),int(len(L)/2)] and j in [0,int(len(ang)/3),int(len(ang)/2),len_ang-1]:
                    plt.figure()
                    plt.loglog(l,spec[j],label='L=%d, ang=%f'%(L[i],ang[j]/np.pi))
                    x=np.exp(np.linspace(np.log(min(l)),np.log(max(l)),300))
                    spec_i=splev(x , result2[j], ext=2) #ext=2: error if value outside of interpolation range
                    plt.loglog(x,spec_i,"ro")
                    plt.legend()
                    plt.savefig('/afs/mpa/temp/vboehm/spectra/interp_plots/bispec_interp_%s_%d_%d.png'%(config,i,j))
                    plt.close()

        result+=[result2]

    return result


"""Settings to chosse correct Bispectrum"""
for red_bin in ['0']:

    cross_bias  = True
    kkg         = True

    LSST        = True

    tag         = 'reverse_int_no_cut'

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = True
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662
    post_born   = False
    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True

    #number of redshift bins
    bin_num     = 50

    #sampling in L/l and angle


    #ell range (for L and l)
    L_min       = 100.
    L_max       = 3000.

    l_min       = 1
    l_max       = 8000.

    k_min       = None#1e-4
    k_max       = None#100.

    fit_z_max   = 1.5

    #tag for L-sampling
    ell_type    ="linlog_halfang"

    Delta_theta = 1e-4

    nl          = True

    cparams     = C.Planck2015_TTlowPlensing


    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'
    #path, where to store results
    path            = "/afs/mpa/temp/vboehm/spectra/"

    "---end settings---"

    for red_bin in ['0']:

        if LSST:
            dn_filename = 'dndz_LSST_i27_SN5_3y'


    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'

    if kkg:
        config = 'kkg_%s'%ell_type
    if LSST:
        config+='bin_%s_%s'%(red_bin,dn_filename)
    else:
        config+='no_binning'

    config+=spectrum_config

    if B_fit:
        config+="_Bfit"

    config +="_"+cparams[0]['name']

    config+=tag

    print "config: %s"%config

    #path, where to store results
    path            = "/afs/mpa/temp/vboehm/spectra/"

    filename=path+"ell_%s_Lmin%d_Lmax%d_lmax%d_%.0e_%s.pkl"%(ell_type,L_min,L_max,l_max,Delta_theta,tag)


    L,l,ang=pickle.load(open(filename))

    len_L = len(L)
    len_l = len(l)
    len_ang = len(ang)

    len_bi = len_L*len_l*len_ang

    if post_born:
        config +='_postBorn_sum'

    path = "/afs/mpa/temp/vboehm/spectra/cross_bias_spectra/"
    loadfile = path+"bispec_phi_%s_Lmin%d-Lmax%d-lmax%d-lenBi%d"%(config,L_min,L_max-1,l_max, len_bi)
    if cross_bias:
        loadfile+='_4bias'



    print 'loading bispectrum ', loadfile

    bi_phi=np.load(loadfile+'.npy')

    splines = bispec_interp(bi_phi, L, l , ang, len_l, len_ang, len_L, config)


    dumpfile=path+'bispec_interp_%s_ll.pkl'%config

    pickle.dump([ang,L,l,splines],open(dumpfile,'w'))
    print "dumped to %s"%dumpfile
