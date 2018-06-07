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
from scipy.interpolate import interp1d
import pickle
from copy import deepcopy

#import matplotlib.pyplot as plt

from classy import Class
import Cosmology as C
from N32biasIntegrals import skew
from Constants import LIGHT_SPEED

from Bispectra import Bispectra


def get_triangles(ell_type,Lmin,Lmax,lmin,lmax,len_L,len_low_L,len_l,len_ang,path,Delta_theta=0):

    filename=path+"ells/ell_ang_%s_Lmin%d_Lmax%d_lmin%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,Lmin,Lmax,lmin,lmax,len_L,len_l,len_ang,Delta_theta)

    if ell_type=="full":

        La      = np.linspace(L_min,100,len_low_L)
        Lb      = np.exp(np.linspace(np.log(100),np.log(L_max),len_L-len(La)+1))[1:]
        L       = np.append(La,Lb)

        la      = L
        lb      = np.exp(np.linspace(np.log(L_max),np.log(l_max),21))[1:]
        l       = np.append(la,lb)
        assert(len(l)==len_l)

    elif ell_type=='equilat':
        assert(len_L>150)
        len_side=len_L
        L       = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_side))
        l       = None
    elif ell_type=='folded':
        assert(len_L>150)
        len_side=len_L
        L       = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_side))
        l       = 0.5*L

    if ell_type=='full':
        theta   = np.linspace(Delta_theta,2*np.pi-Delta_theta, len_ang)
    if ell_type=='equilat':
        theta   = np.asarray([np.pi/3.]*len_side)
    if ell_type=='folded':
        theta   = np.asarray([0.]*len_side)

    print filename
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
    elif ell_type=='folded':
        for i in range(len_side):
            l1= L[i]
            l3= l[i]
            l2= l[i]
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

    angs = (np.asarray(ang12),np.asarray(ang23),np.asarray(ang31))
    angmu = np.array(angmu)
    ls = (np.asarray(ell1),np.asarray(ell2),np.asarray(ell3))

    return ls, angs, angmu, filename


#def q(chi,chi_lim,n):


#def gal_lens(chi,z,chimax,cosmo):
#    def q(x,xmax):
#      simps((xmax-x)/xmax)
#    kernel =(1+z)*chi*q(chi)
#    factor = cosmo.cmb_prefac
#    return kernel*factor

def CMB_lens(chi,z,chicmb,cosmo):
    kernel =(1+z)*chi*(chicmb-chi)/chicmb
    factor = cosmo.cmb_prefac
    return kernel*factor

##all in kappa
"""---- Choose your settings here ---"""
if __name__ == "__main__":

    "---begin settings---"

    tag         = 'test'

    ell_type    = 'equilat'#'equilat','folded'

    cparams     = C.Pratten
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662)
    post_born   = False

    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = False
    fit_z_max   = 5.
    nl          = True
    #number of redshift bins
    bin_num     = 100
    z_min       = 1e-4

    #sampling in L/l and angle
    len_L       = 200
    len_l       = len_L+20
    len_ang     = len_L

    #ell range (for L and l)
    L_min       = 10.
    L_max       = 10000.
    len_low_L   = 20

    l_min       = L_min
    l_max       = 8000.

    Delta_theta = 0.

    k_min       = 1e-4#times three for lens planes
    k_max       = 100.
    #k-range1: 0.0105*cparams[1]['h']-42.9*cparams[1]['h']
    #k-range2: 0.0105*cparams[1]['h']-49*cparams[1]['h']


    path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"

    "---end settings---"

    assert(ell_type in ['folded','full','equilat'])

    ls, angs, angmus, ellfile= get_triangles(ell_type,L_min,L_max,l_min,l_max,len_L,len_low_L,len_l,len_ang,path,Delta_theta=Delta_theta)

    params  = deepcopy(cparams[1])
    acc = deepcopy(C.acc_1)
    params.update(acc)

    closmo  = Class()
    closmo.set(params)
    closmo.compute()
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    del closmo

    print "z_cmb: %f"%z_cmb

    zmax  = z_cmb-0.0001
    za    = np.exp(np.linspace(np.log(z_min),np.log(10.),int(3*bin_num/4)))
    zb    = np.linspace(10.,zmax,int(bin_num/4+1))[1::]
    z     = np.append(za,zb)
    print(z)
    assert(len(z)==bin_num)


    data    = C.CosmoData(params,z)

    chi     = data.chi(z)
    chicmb  = data.chi(z_cmb)

    config  = tag+"_"+ell_type+"_"+cparams[0]['name']

    kernels = (CMB_lens(chi,z,chicmb,data),None,None)

    print "config: %s"%config

    bs   = Bispectra(params,data,ls[0],ls[1],ls[2],z,chi,kernels,config,angs[0],angs[1],angs[2],path, nl,B_fit,k_min,k_max,fit_z_max,ft='SC')

    bs()

    print(ellfile)
    print(bs.filename)


##TODO: check everything beneath
#    if post_born:
#        import CAMB_postborn as postborn
#        print 'computing post Born corrections...'
#        assert(kkk or kkg)
#
#        config +='_postBorn'
#
##        if bs.set_stage==False:
##            bs.set_up()
#        k_min   = bs.kmin
#        k_max   = bs.kmax
#        PBB     = postborn.PostBorn_Bispec(params,k_min,k_max,kkg,dndz,norm,zmaxint=zmax)
#
#
#        if kkg:
#            try:
#                bi_kkg_sum  = np.load(bs.filename+"_post_born_sum.npy")
#                bi_kkg      = np.load(bs.filename+"_post_born.npy")
#            except:
#                prefac      = 16./(3.*data.Omega_m0*data.H_0**2)*LIGHT_SPEED**2
#                #L is associated wit galaxy leg in bias, in CAMBPostBorn it's L3
#                bi_kkg      = PBB.bi_born_cross(ell2,ell3,ell1,prefac,sym=bs.sym)
#                bi_kkg_sum  = bi_kkg+bs.bi_phi
#                np.save(bs.filename+"_post_born.npy",bi_kkg)
#                np.save(bs.filename+"_post_born_sum.npy",bi_kkg_sum)
#
#            bi_phi = bi_kkg_sum
#        else:
#            bi_post  = (PBB.bi_born(ell1,ell2,ell3)*8./(ell1*ell2*ell3)**2)
#            np.save(bs.filename+"_post_born.npy",bi_post)
#            np.save(bs.filename+"_post_born_sum.npy",bi_post+bs.bi_phi)
#            bi_phi = bi_post+bs.bi_phi




