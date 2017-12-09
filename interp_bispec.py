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

def bispec_interp(bispec, ell, cos13, len_l, len_ang, len_L, plot=True):
    """interpolates the bispectrum computed in Spectra.py
    * bispec: Bispectrum(ell[0::3], ell[1::3],ell[2::3])
    * ell: arguments to bispec
    * ang13: angle between L and l
    * bin size: len(ell)
    * sample1d: length of unique L array    
    """
    # |-L| 
    L = ell[0::3]

    # |l|
    l = ell[2::3]
    
    result=[]
    bin_size=len_l*len_ang
    
    
    # for every L interpolate over l and angle go through every L and interpolate for for every L
    for i in np.arange(len_L):
        i  = np.int(i)
        l_ = l[i*bin_size:(i+1)*bin_size]
        ang = cos13[i*bin_size:(i+1)*bin_size] #angle between vec L and vec l 
        spec=bispec[i*bin_size:(i+1)*bin_size]
        l_=np.reshape(l_,(len_l,len_ang))
        ang=np.reshape(ang,(len_l,len_ang))
        spec=np.reshape(spec,(len_l,len_ang))

            
        result2=[]
        #for every fixed l interpolate in angle
        for j in range(len_l):
            result2+=[splrep(ang[j], spec[j], k=1)]
            if plot:
                if i in [0,10,20] and j in [0,10,20]:
                    plt.figure()
                    plt.plot(ang[j],spec[j])
                    x=np.linspace(min(ang[j]),max(ang[j]),200)
                    spec_i=splev(x , result2[j], ext=2) #ext=2: error if value outside of interpolation range
                    plt.plot(x,spec_i,"ro")
                    plt.savefig('/afs/mpa/temp/vboehm/spectra/interp_plots/bispec_interp_%s_%d_%d.png'%(config,i,j))
            
            assert(np.allclose(ang[j],ang[j-1]))
        result+=[result2]
        
    return ang[0], np.unique(L), np.unique(l), result


"""Settings to chosse correct Bispectrum"""
for red_bin in ['0','1','2']:
    
    
    LSST        = False 
    dn_filename = 'dndz_LSST_i27_SN5_3y'
    cross_bias  = True
    #red_bin     = '0'
    #choose Cosmology (see Cosmology module)
    params      = C.Planck2015_TTlowPlensing
    
    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
    
    #sampling in L/l and angle
    len_L       = 163
    len_l       = 163
    len_ang     = 163
    
    len_bi      = len_L*len_l*len_ang
    
    #ell range (for L and l)
    L_min       = 1.
    L_max       = 10000.
    
    l_min       = 1.
    l_max       = 10000.
    
    k_min       = 1e-4
    k_max       = 100.
    
    z_min       = 1e-4
    
    post_born   = False
    
    spectra_configs=['_nlPS']
    
    
    ell_type    ="linlog_halfang"
    
    Delta_theta = 1e-2
    
    nl          = True
    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'
        
    path = "/afs/mpa/temp/vboehm/spectra/"
    
    filename=path+"ell_%s_Lmin%d_Lmax%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_max,len_L,len_l,len_ang,Delta_theta)
    filename_ang=path+"ang_%s_Lmin%d_Lmax%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_max,len_L,len_l,len_ang,Delta_theta)
            
    ell=pickle.load(open(filename))
    angles=pickle.load(open(filename_ang))
    angmu=angles[3]
    del angles
    
    config = 'kkg_%s'%ell_type
    
    if LSST:
        config+='bin_%s_%s'%(red_bin,dn_filename)
    else:
        config+='no_binning'
    
    config+=spectrum_config
    if B_fit:
        config+="_Bfit"
        config +="_"+params[0]['name']
        
    path = "/afs/mpa/temp/vboehm/spectra/cross_bias_spectra/"
    loadfile = path+"bispec_phi_%s_Lmin%d-Lmax%d-lmax%d-lenBi%d"%(config,L_min,L_max,l_max,len_bi)
    
    if cross_bias:
        loadfile+='_4bias'
    
    bi_phi=np.load(loadfile+'.npy')
    
    x, Ls,ls,splines = bispec_interp(bi_phi, ell, angmu, len_l, len_ang, len_L, config)
    if cross_bias:
        config+='_4bias'
        
    dumpfile=path+'bispec_interp_%s_mu.pkl'%config
    
    pickle.dump([x,Ls,ls,splines],open(dumpfile,'w'))
    print "dumped to %s"%dumpfile
