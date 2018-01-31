# -*- coding: utf-8 -*-
"""
Created on 12.02.2015

@author: Vanessa Boehm

Spectra.py:

    * class Bispectra
    * functions for computation of dm power spectrum
"""
from __future__ import division
import numpy as np


from scipy.integrate import simps
from scipy.interpolate import interp1d, splev
import pickle
from copy import deepcopy

import matplotlib.pyplot as pl

from classy import Class
import Cosmology as C
from N32biasIntegrals import I0, I2, skew
from Constants import LIGHT_SPEED

from Bispectra import Bispectra



"""---- Choose you settings here ---"""
if __name__ == "__main__":

    "---begin settings---"

    tag         = 'sim_comp_postmerge2'

    #type of bispectrum
    kkg         = False
    kgg         = False
    kkk         = True

    #triangle configuration
    ell_type    ='full'#'equilat','folded'

    #compute beta integrals?
    integrals   = True

    skewness    = True
    FWHMs       = [0.5,1.,2.,3.,4.,5.,8.,10.]

    #use LSST like redshift bins
    LSST        = False

    #if kkg or kgg: symmetrize over positions of k/g
    sym         = False

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    #Limber      = False

    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662)
    post_born   = True

    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
    fit_z_max   = 1.5

    #number of redshift bins
    bin_num     = 80
    z_min       = 1e-3

    #sampling in L/l and angle
    len_L       = 100
    len_l       = len_L+20
    len_ang     = 100

    #ell range (for L and l)
    L_min       = 1. #set to 2
    L_max       = 3000.

    l_min       = L_min
    l_max       = 8000.


    Delta_theta = 1e-4

    nl          = True
    cparams     = C.SimulationCosmology#C.Planck2015_TTlowPlensing#

    k_min       = 0.0105*cparams[1]['h']
    k_max       = 42.9*cparams[1]['h']
    #k-range1: 0.0105*cparams[1]['h']-42.9*cparams[1]['h']
    #k-range2: 0.0105*cparams[1]['h']-49*cparams[1]['h']

    #path, where to store results
    path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"


    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'

    if ell_type=='equilat':
        len_side= 250

    "---end settings---"

    assert(kgg+kkk+kkg==1)
    assert(ell_type in ['folded','full','equilat'])


    params  = deepcopy(cparams[1])
    closmo  = Class()
    closmo.set(params)
    closmo.compute()
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    del closmo

    print "z_cmb: %f"%z_cmb

    z_a     = np.exp(np.linspace(np.log(z_min),np.log(100.),70,endpoint=False))
    z_b     = np.linspace(100.,z_cmb-0.001,10)
    z       = np.append(z_a,z_b)
    assert(len(z)==bin_num)

    if kkg or kgg:

        for red_bin in ['0']:

            if LSST:

                dn_filename = 'dndz_LSST_i27_SN5_3y'

                if red_bin!='None':
                    bins,big_grid,res   = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
                    mbin                = bins[int(red_bin)]
                    zbin                = big_grid
                    nbin                = res[int(red_bin)]
                    dndz    = interp1d(zbin, nbin, kind='linear',bounds_error=False,fill_value=0.)
                    print 'using z-bin', mbin
                else:
                    gz, dgn = pickle.load(open(dn_filename+'tot_extrapolated.pkl','r'))
                    dndz    = interp1d(gz, dgn, kind='linear',bounds_error=False,fill_value=0.)

                z       = np.linspace(min(mbin[0][0]-0.3,z_min),mbin[0][1]+0.3,bin_num)
                bias    = z+1.
                norm    = simps(dndz(z),z)

            else:
                z0      = 1./3.
                dndz    = (z/z0)**2*np.exp(-z/z0)
                dndz    = interp1d(z,dndz,kind='slinear',fill_value=0.,bounds_error=False)
                bias    = 1.
                norm    = simps(dndz(z),z)

    else:
        dndz    = None
        norm    = None
        bias    = None

    print 'z-range: ', min(z), max(z)
    print 'norm: ', norm

        #cosmo dependent functions
    data    = C.CosmoData(params,z)

    filename=path+"ell_ang_%s_Lmin%d_Lmax%d_lmin%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_min,l_max,len_L,len_l,len_ang,Delta_theta)

    if ell_type=="full":
        #L = |-L|, equally spaced in lin at low L and in log at high L
        L       = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_L))
        la      = L
        lb      = np.exp(np.linspace(np.log(L_max),np.log(l_max),21))[1:]
        l       = np.append(la,lb)
        assert(len(l)==len_l)

    elif ell_type=='equilat':
        assert(len_side>150)
        L       = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_side))
        l       = None

    if ell_type=='full':
        theta   = np.linspace(Delta_theta,2*np.pi-Delta_theta, len_ang)
    if ell_type=='equilat':
        theta   = np.asarray([np.pi/3.]*len_side)
    if ell_type=='folded':
        theta   = np.asarray([0.]*len_side)


    pickle.dump([L,l,theta],open(filename, 'w'))

    cosmu   = np.cos(theta) #Ldotl/Ll or -l1dotl3/l1/l3 (l1+l2+l3=0) (angle used in beta Integrals)

    ang31=[]
    ang12=[]
    ang23=[]
    angmu=[]
    ell1 =[]
    ell2 =[]
    ell3 =[]

    sqrt=np.sqrt
            #all combinations of the two sides and the angles

    if ell_type=='equilat':
        for i in range(len_side):
            l1= L[i]
            l3= L[i]
            l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
            ell1+=[l1]
            ell2+=[l2]
            ell3+=[l3]
            ang31+=[-cosmu[i]]
            ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
            ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
            angmu+=[theta[i]]
    elif ell_type=='full':
        for i in range(len_L):
            for k in range(len_l):
                for j in range(len_ang):
                    l1= L[i]
                    l3= l[k]
                    l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[j])
                    ell1+=[l1]
                    ell2+=[l2]
                    ell3+=[l3]
                    ang31+=[-cosmu[j]]
                    ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                    ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                    angmu+=[theta[j]]
            #array of length 3*number of triangles

    ang12=np.array(ang12)
    ang23=np.array(ang23)
    ang31=np.array(ang31)
    angmu=np.array(angmu)
    ell1=np.array(ell1)
    ell2=np.array(ell2)
    ell3=np.array(ell3)

    if kkg:
        config = 'kkg_%s'%ell_type
    elif kgg:
        config = 'kgg_%s'%ell_type
    else:
        config = 'kkk_%s'%ell_type

    if LSST:
        config+='LSST_bin_%s_%s'%(red_bin,dn_filename)
    else:
        config+='analytic_red_dis'

    if sym:
        config+='_sym'

    config+=spectrum_config

    if B_fit:
        config+="_Bfit"

    config +="_"+cparams[0]['name']

    config+=tag

    print "config: %s"%config

    bs   = Bispectra(params,data,ell1,ell2,ell3,z,config,ang12,ang23,ang31,path,z_cmb, bias, nl,B_fit,kkg, kgg, kkk,dndz, norm,k_min,k_max,sym,fit_z_max)

    bs()



    if integrals:
        Int0 = I0(bs.bi_phi, L, l, theta, len_l, len_L,len_ang)

        Int2 = I2(bs.bi_phi, L, l, theta, len_l, len_L,len_ang)

        pickle.dump([params,L,Int0,Int2],open(path+'integrals/I0I1I2%s.pkl'%(config),'w'))
        print path+'integrals/I0I1I2%s.pkl'%(config)
    if skewness:
        res=[]
        for FWHM in FWHMs:
            res+=[skew(bs.bi_phi, FWHM, L, l, ell2, theta, len_l, len_L,len_ang,kappa=True)]
        print res
        pickle.dump([FWHMs,skew],open(path+'skewness_%s.pkl'%(config),'w'))


#TODO: check everything beneath
    if post_born:
        import CAMB_postborn_old as postborn
        print 'computing post Born corrections...'
        assert(kkk or kkg)

        config +='_postBorn'

#        if bs.set_stage==False:
#            bs.set_up()
        k_min   = bs.kmin
        k_max   = bs.kmax
        PBB     = postborn.PostBorn_Bispec(params,k_min,k_max)#,kkg,dndz,norm)


        if kkg:
            try:
                bi_kkg_sum  = np.load(bs.filename+"_post_born_sum.npy")
                bi_kkg      = np.load(bs.filename+"_post_born.npy")
            except:
                prefac      = 16./(3.*data.Omega_m0*data.H_0**2)*LIGHT_SPEED**2
                #L is associated wit galaxy leg in bias, in CAMBPostBorn it's L3
                bi_kkg      = PBB.bi_born_cross(ell2,ell3,ell1,prefac,sym=bs.sym)
                bi_kkg_sum  = bi_kkg+bs.bi_phi
                np.save(bs.filename+"_post_born.npy",bi_kkg)
                np.save(bs.filename+"_post_born_sum.npy",bi_kkg_sum)

            bi_phi = bi_kkg_sum
        else:
            bi_post  = (PBB.bi_born(ell1,ell2,ell3)*8./(ell1*ell2*ell3)**2)
            np.save(bs.filename+"_post_born.npy",bi_post)
            np.save(bs.filename+"_post_born_sum.npy",bi_post+bs.bi_phi)
            bi_phi = bi_post+bs.bi_phi

        if integrals:
            Int0 = I0(bi_phi, L, l, theta, len_l, len_L,len_ang)

            Int2 = I2(bi_phi, L, l, theta, len_l, len_L,len_ang)

            pickle.dump([params,L,Int0,Int2],open(path+'integrals/I0I1I2%s.pkl'%(config),'w'))
            print path+'integrals/I0I1I2%s.pkl'%(config)
        if skewness:
            res=[]
            for FWHM in FWHMs:
                res+=[skew(bi_phi, FWHM, L, l, ell2, theta, len_l, len_L,len_ang,kappa=True)]
            print res
            pickle.dump([FWHMs,skew],open(path+'skewness_%s.pkl'%(config),'w'))

        del bs
        try:
            del bi_phi
        except:
            pass

