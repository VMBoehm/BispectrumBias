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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

from classy import Class
import Cosmology as C
from N32biasIntegrals import I0, I2
import CAMB_postborn as postborn
from Constants import LIGHT_SPEED

from Bispectra import Bispectra



"""---- Choose you settings here ---"""
if __name__ == "__main__":

    "---begin settings---"

    tag         = 'laptop_test1'

    #type of bispectrum
    kkg         = False
    kgg         = False
    kkk         = True

    #triangle configuration
    ell_type    ='full'#'equilat','folded'

    #compute beta integrals?
    integrals   = True

    #use LSST like redshift bins
    LSST        = False

    #if kkg or kgg: symmetrize over positions of k/g
    sym         = False

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = True

    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662)
    post_born   = False

    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = False
    fit_z_max   = 1.5

    #number of redshift bins
    bin_num     = 100
    z_min       = 1e-4

    #sampling in L/l and angle
    len_L       = 80
    len_l       = 200
    len_ang     = 200

    #ell range (for L and l)
    L_min       = 100.
    L_max       = 3000.

    l_min       = 1
    l_max       = 8000.

    k_min       = None#1e-4
    k_max       = None#100.


    Delta_theta = 1e-4

    nl          = True

    cparams     = C.SimulationCosmology#Planck2015_TTlowPlensing

    #path, where to store results
    path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"


    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'

    if ell_type=='equilat':
        len_side= 250
    if ell_type=='folded':
        len_side= 250


    "---end settings---"

    assert(kgg+kkk+kkg==1)
    assert(ell_type in ['folded','all','equilat'])


    params  = deepcopy(cparams[1])
    closmo  = Class()
    closmo.set(params)
    closmo.compute()
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    del closmo

    print "z_cmb: %f"%z_cmb

    z       = np.exp(np.linspace(np.log(z_min),np.log(z_cmb),bin_num))

    # avoid numerical inaccuracies
    if z[-1]!=z_cmb:
        z[-1]=z_cmb

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


    filename=path+"ell_ang_%s_Lmin%d_Lmax%d_lmin%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_max,len_L,len_l,len_ang,Delta_theta)

    if ell_type=="full":
        #L = |-L|, equally spaced in lin at low L and in log at high L
        L         = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_L))
        la        = np.linspace(l_min,20,20,endpoint=False)
        lb        = np.linspace(20,L_max,len_l-40,endpoint=False)
        lc        = np.exp(np.linspace(np.log(L_max),np.log(l_max),20))
        l1        = np.append(la,lb)
        l         = np.append(l1,lc)
        assert(len(l)==len_l)

    elif ell_type=='equilat':
        assert(len_side>150)
        La        = np.arange(L_min,150)
        Lb        = np.ceil(np.exp(np.linspace(np.log(150),np.log(L_max),len_side-len(La),)))
        L         = np.append(La,Lb)
    elif ell_type=='folded':
        assert(len_side>150)
        La        = np.arange(L_min,150)
        Lb        = np.ceil(np.exp(np.linspace(np.log(150),np.log(L_max),len_side-len(La))))

        L         = np.append(La,Lb)
        l         = np.append(La,Lb)*0.5


    theta   = np.linspace(Delta_theta,np.pi-Delta_theta, len_ang)
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
    ell  =[]
    sqrt=np.sqrt
            #all combinations of the two sides and the angles

    if ell_type=='equilat':
        for i in range(len_side):
                l1= L[i]
                l3= L[i]
                l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
                ell+=[l1]+[l2]+[l3]
                ang31+=[-cosmu[i]]
                ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                angmu+=[theta[i]]
    elif ell_type=='folded':
        for i in range(len_side):
                l1= L[i]
                l3= l[i]
                l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
                ell+=[l1]+[l2]+[l3]
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
                    ell+=[l1]+[l2]+[l3]
                    ang31+=[-cosmu[j]]
                    ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                    ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                    angmu+=[theta[j]]
            #array of length 3*number of triangles

    ang12=np.array(ang12)
    ang23=np.array(ang23)
    ang31=np.array(ang31)
    angmu=np.array(angmu)
    ell=np.asarray(ell)


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


    bs   = Bispectra(params,data,ell,z,config,ang12,ang23,ang31,path,z_cmb, bias, nl,B_fit,kkg, kgg, kkk,dndz, norm,k_min,k_max,sym,fit_z_max)

    bs()

#TODO: check everything beneath

    if integrals:
        Int0 = I0(bs.bi_phi, bs.ell, angmu, len_L, len_l, len_ang)

        Int2 = I2(bs.bi_phi, bs.ell, angmu ,len_L, len_l, len_ang)

        pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s.pkl'%(config),'w'))

    if post_born:
        print 'computing post Born corrections...'
        assert(kkg or kkk)

        config +='_postBorn'

        if bs.set_stage==False:
            bs.set_up()
        k_min   = bs.kmin
        k_max   = bs.kmax
        PBB     = postborn.PostBorn_Bispec(params,k_min,k_max,kkg,dndz,norm)

        if kkg:
            try:
                bi_kkg_sum  = np.load(bs.filename+"_post_born_sum.npy")
                bi_kkg      = np.load(bs.filename+"_post_born.npy")
            except:
                prefac      = 16./(3.*data.Omega_m0*data.H_0**2)*LIGHT_SPEED**2
                #L is associated wit galaxy leg in bias, in CAMBPostBorn it's L3
                bi_kkg      = PBB.bi_born_cross(ell[1::3],ell[2::3],ell[0::3],prefac,sym=bs.sym)
                bi_kkg_sum  = bi_kkg+bs.bi_phi
                np.save(bs.filename+"_post_born.npy",bi_kkg)
                np.save(bs.filename+"_post_born_sum.npy",bi_kkg_sum)

            bi_phi = bi_kkg_sum
        else:
            bi_post  = (PBB.bi_born(ell[0::3],ell[1::3],ell[0::3])*8./(ell[0::3]*ell[1::3]*ell[2::3])**2)
            np.save(bs.filename+"_post_born.npy",bi_post)
            np.save(bs.filename+"_post_born_sum.npy",bi_post+bs.bi_phi)
            bi_phi = bi_post+bs.bi_phi

        if integrals:
            Int0 = I0(bi_phi, bs.ell, angmu ,len_L, len_l, len_ang)

            Int2 = I2(bi_phi, bs.ell, angmu ,len_L, len_l, len_ang)

            L    = np.unique(ell[0::3])

            pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s.pkl'%(config),'w'))
            Int0 = I0(bi_kkg, bs.ell, angmu ,len_L, len_l, len_ang)

            Int2 = I2(bi_kkg, bs.ell, angmu ,len_L, len_l, len_ang)

            pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s_only.pkl'%(config),'w'))

        del bs
        try:
            del bi_phi
        except:
            pass










