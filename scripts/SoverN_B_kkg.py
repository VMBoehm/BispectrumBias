# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:29:14 2016
SoverN_Bphi.py
Computes S/N of B_phi in flat sky for simple Gaussian Noise
@author: VBoehm
"""

import pylab as plt
import pickle
from scipy.integrate import simps
import numpy as np
from scipy.interpolate import interp1d
import Cosmology as Cosmo

def SN_integral(bispec,var_len, var_gal, var_xx, Ls, ls, Lls, ang, bin_size, sample1d, min_index, max_index):
    # for every L interpolate over l and angle and integrate over intrepolated 2d function
    L_integrand=[]
    lmin=Ls[min_index]
    lmax=Ls[max_index]
    print "integrate L from ", lmin, "to ", lmax
    for i in np.arange(min_index,max_index+1):
        L_      = Ls[i]
        spec    = bispec[i*bin_size:(i+1)*bin_size]
        Ll_     = Lls[i*bin_size:(i+1)*bin_size]

        integrand = []

        for j in np.arange(i,max_index+1): #start at l=L
            l_const     = ls[j]
            spec_int    = spec[j*sample1d:(j+1)*sample1d]

            Lli         = Ll_[j*sample1d:(j+1)*sample1d]

            index       = np.where((Lli<=lmax)*(Lli>=l_const))

            spec_int    = spec_int[index] #restrict
            ang_        = ang[index]
            Ll          = Lli[index]

            fac         = np.ones(len(Ll))


            if j==i:
                 fac*=2.
                 fac[np.where(np.isclose(ang_,np.pi/3.))]=6.
            if len(Ll)>=1:
                num   = (spec_int*2)**2
                denom = fac*var_lens(L_)*var_lens(l_const)*var_gal(Ll)#extrapolate
                integrand+=[simps(num/denom,ang_)]
            else:
                integrand+=[0]

        L_integrand += [simps(integrand*ls[i:max_index+1],ls[i:max_index+1])]

    res = simps(Ls[min_index:max_index+1]*L_integrand,Ls[min_index:max_index+1])
    print max(Ls[min_index:max_index+1])
    return lmin, lmax, res


if __name__ == "__main__":

    LSST        = False
    red_bin     = '0'
    params      = Cosmo.ToshiyaComparison
    tag         = params[0]['name']+'_nl'

    if LSST:
        dn_filename = 'dndz_LSST_i27_SN5_3y'
    else:
        dn_filename = 'red_dis_func'
        red_bin     = 'None'

    ell_min     = 1.
    ell_max     = 10000.
    len_L       = 163
    len_ang     = 163
    Delta_theta = 1e-2

    La          = np.linspace(ell_min,50,48,endpoint=False)
    Lb          = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
    side1       = np.append(La,Lb)

    fsky        = 1.

    theta       = np.linspace(Delta_theta,np.pi, len_ang)

    #Parameter,cl_unl,cl_len   = pickle.load(open('../class_outputs/class_cls_%s.pkl'%tag,'r'))

    ll,var_lens,var_gal,cl_xx = pickle.load(open('Gaussian_variances_CMB-S4_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'r'))
    print 'Gaussian_variances_CMB-S4_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename)
    print 'Gaussian_variances_CMB-S4_binNone_Toshiya_nl_red_dis_func.pkl'
    b_kkg_ns       = np.load("bispec_phi_kkg_g_bin0linlog_halfang_lnPsToshiyaSettings_Bfit_Toshiya_Lmin1-Lmax10000-lmax10000-lenBi4330747.npy")
    b_kkg_s       = np.load("bispec_phi_kkg_g_bin0linlog_halfang_lnPsToshiyaSettings_Bfit_Toshiya_Lmin1-Lmax10000-lmax10000-lenBi4330747_sym.npy")
    Ll          = np.asarray(np.load(open('Ll_file_linlog_halfang_1e+00_10000_lenL163_lenang163_1e-02.pkl','r')))

    var_gal     = interp1d(ll,var_gal, bounds_error=False, fill_value=np.inf)
    var_lens    = interp1d(ll,var_lens, bounds_error=False, fill_value=np.inf)
    var_xx      = interp1d(ll,cl_xx, bounds_error=False, fill_value=np.inf)

#    min_L       = []
#    SN          = []
#    index_max   = 159
#
#    for index_min in np.arange(2,159,10):
#        print index_min, index_max
#        minL_, maxL_, SN_ = SN_integral(b_kkg, var_lens, var_gal, var_xx, side1, side1, Ll, theta, len_L**2, len_L, index_min, index_max)
#        min_L     +=[minL_]
#        SN        +=[SN_*fsky/(2*np.pi**2)]
#
#
#        plt.semilogx(min_L, np.asarray(min_L)*SN/max(SN) ,marker="o")
#    plt.legend()
#    plt.xlabel(r'$L_{min}$')
#    plt.ylabel("S/N")
#    plt.savefig("S_over_N_Toshiya_B_kkg_lmax%d_thetaFWHMarcmin30_noiseUkArcmin07_fsky%.1f.pdf"%(maxL_,fsky), bbox_inches="tight")
#    plt.show()


    index_min   = 0

    labels=['$B^{kkg}$ non sym']#,'$B^{kkg}$ sym']
    ii=0
    for b_kkg in [b_kkg_ns]:#, b_kkg_s]:
        max_L       = []
        SN          = []
        for index_max in np.arange(20,159,3):
            minL_, maxL_, SN_ = SN_integral(b_kkg, var_lens, var_gal, var_xx, side1, side1, Ll, theta, len_L**2, len_L, index_min, index_max)
            max_L     +=[side1[index_max]]
            SN        +=[SN_*fsky/(2*np.pi**2)]
        plt.plot(max_L, np.sqrt(SN),label=labels[ii])
        ii+=1
    plt.xlim(0,3000)
    plt.ylim(0,160)
    plt.legend(loc='best')
    plt.xlabel(r'$L_{\mathrm{max}}$')
    plt.ylabel("S/N")
    plt.savefig("S_over_N_Toshiya_B_kkg_lmin%d_thetaFWHMarcmin30_noiseUkArcmin07_fsky%.1f.pdf"%(minL_,fsky), bbox_inches="tight")
    plt.show()
