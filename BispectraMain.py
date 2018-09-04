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
import warnings




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
    elif ell_type=='squeezed':
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
    if ell_type=='squeezed':
        theta   = np.asarray([2.*np.pi-Delta_theta]*len_side)

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
    elif ell_type=='squeezed':
        ell_type+='_ang%s'%str(Delta_theta)
        for i in range(len_side):
            l1= L[i]
            l3= L[i]
            #l2= 0. #l
            l2 = sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
            ell1+=[l1]
            ell2+=[l2]
            ell3+=[l3]
            ang31+=[-cosmu[i]]
            ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
            ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
#            ang12+=[0.]
#            ang23+=[0.]
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


#redshift distribution used in astro-ph/0310125v4
def p_z(cosmo,z0=0.5, nbar=100., z_s=None):
    def p_chi(z):
      return nbar*z**2/(2.*z0**3)*np.exp(-z/z0)*cosmo.dzdchi(z)
    return p_chi


def p_delta(cosmo,z_s):
    def p_chi(z):
      w = np.zeros(len(z))
      w[np.isclose(z,z_s)]=cosmo.dzdchi(z_s)
      assert(np.any(w is not 0.))
      return w

    return p_chi



def gal_lens(zrange,cosmo, p_chi=None):

    chimin, chimax = (cosmo.chi(zrange[0]),cosmo.chi(zrange[1]))
    q = []
    chi_ = np.linspace(0,chimax,int(chimax)*20)

    for cchi in chi_:
        x_= np.linspace(max(chimin,cchi),chimax,max(int(chimax-max(chimin,cchi))*10,200))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrand = p_chi(data.zchi(x_))*(x_-cchi)/x_
            integrand[x_==0.]=0.
            q+=[simps(integrand,x_)]
    q[-1]=0.
    q = interp1d(chi_,q,bounds_error=False, fill_value=0.)

    chi_ = np.linspace(chimin,chimax,int(chimax-chimin)*10)
    norm = simps(p_chi(data.zchi(chi_)),chi_)
    print(norm)


    def kernel(x,z):

      w = np.ones(x.shape)
      w[x>chimax] =0.
      w[x==0.] =0.
      res = w*x*q(x)*(1.+z)

      return res*cosmo.lens_prefac/norm

    return kernel


def CMB_lens(chicmb,cosmo):
    def kernel(x,z,chimax=None):
      if chimax is None:
        chimax=chicmb
      w = np.ones(x.shape)
      w[x>chimax]==0.
      return (1+z)*x*w*(chimax-x)/chimax*cosmo.lens_prefac
    return kernel


def simple_bias(z):
    return 1.+z


def dNdz_LSST(bin_num,dn_filename = 'dndz_LSST_i27_SN5_3y'):
    if bin_num is "all":
      zbin, nbin = pickle.load(open(dn_filename+'tot_extrapolated.pkl','r'))
      norm                = simps(nbin,zbin)
      mbin                = 'None'
    else:
      bins,big_grid,res   = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
      mbin                = bins[bin_num]
      zbin                = big_grid
      nbin                = res[bin_num]
      norm                = simps(nbin,zbin)
    dndz                = interp1d(zbin, nbin/norm, kind='linear',bounds_error=False,fill_value=0.)
    print 'using z-bin', mbin, 'norm', norm
    return dndz



def gal_clus(dNdz,b,cosmo,bin_num):
    p_z=dNdz(bin_num)
    def kernel(x,z):
      return b(z)*p_z(z)*cosmo.dzdchi(z)
    return kernel



##all in kappa
"""---- Choose your settings here ---"""
if __name__ == "__main__":

    "---begin settings---"

    tag         = 'kkg_new'

    ell_type    = 'full'#'equilat','folded'

    cparams     = C.Planck2015
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662)
    post_born   = True

    neutrinos   = False

    cross_bias  = True

    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
    fit_z_max   = 5.
    nl          = True
    #number of redshift bins
    bin_num     = 100
    z_min       = 1e-4 #for squeezed galaxy lens

    #sampling in L/l and angle
    len_L       = 120 #120
    len_l       = len_L+20
    len_ang     = len_L

    #ell range (for L and l)
    L_min       = 1.
    L_max       = 3000.
    len_low_L   = 20

    l_min       = L_min
    l_max       = 8000.

    Delta_theta = 1e-4

    k_min       = 1e-4
    k_max       = 50.

    LSST_bin    = 1

    CLASS_Cls   = False

    path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/"

    "---end settings---"

    assert(ell_type in ['folded','full','equilat','squeezed'])


    ls, angs, angmus, ellfile= get_triangles(ell_type,L_min,L_max,l_min,l_max,len_L,len_low_L,len_l,len_ang,path,Delta_theta=Delta_theta)

    params  = deepcopy(cparams[1])
    acc = deepcopy(C.acc_1)
    params.update(acc)

    if neutrinos:
      assert(cparams[0]['name'] is "JiaGalaxyLens")
      params.update(C.JiaNu)
      tag+='_massive_nus'

    closmo  = Class()
    closmo.set(params)
    closmo.compute()
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    del closmo

    print "z_cmb: %f"%z_cmb

    zmax  = z_cmb-1e-4
    a     = np.linspace(1./(1.+z_min),1./(1.+zmax),bin_num)
    z     = 1./a-1.

    assert(len(z)==bin_num)


    data    = C.CosmoData(params,z)

    chi     = data.chi(z)
    chicmb  = data.chi(z_cmb)

    if LSST_bin is not None:
        tag = tag+"_"+"LSSTbin"+str(LSST_bin)

    config  = tag+"_"+ell_type+"_"+cparams[0]['name']
    if ell_type=='squeezed':
        config  = tag+"_"+ell_type+"_ang"+str(Delta_theta)+"_"+cparams[0]['name']

#### kernels ####
    kernels = (gal_clus(dNdz_LSST,simple_bias,data,LSST_bin),CMB_lens(chicmb,data), CMB_lens(chicmb,data))

    print "config: %s"%config

    bs   = Bispectra(params,data,ls[0],ls[1],ls[2],z,chi,kernels,config,angs[0],angs[1],angs[2],path, nl,B_fit,k_min,k_max,fit_z_max,ft='SC')

    bs()

    print(ellfile)
    print(bs.filename)
    print(bs.filenameCL)

    if CLASS_Cls:
      lens={'output':'tCl sCl, mPk',
      'selection':'dirac',
      'selection_mean': '2.5',
  #     'selection_width' :'0.2,0.2',
      'l_switch_limber':1.,
      #can be lower for linear Pk
      'l_max_lss':6000,
      'P_k_max_1/Mpc': k_max,
      'z_max_pk': max(z)}#,
      #'non_diagonal':2}

      if nl==True:
        lens['non linear']='halofit'

      print(params)
      params.update(lens)
      closmo  = Class()
      closmo.set(params)
      closmo.compute()
      cll= closmo.density_cl()
      ll = cll['ell']
      cls0=(1./4.)*(ll+2.)*(ll+1.)*(ll)*(ll-1.)*cll['ll'][0]
      cls0_=(1./4.)*ll**4*cll['ll'][0]

      np.save(bs.filenameCL+'_CLASS'+'.npy',[ll,cls0,cls0_])



    if post_born:
        import CAMB_postborn as postborn
        print 'computing post Born corrections...'

        PBB     = postborn.PostBorn_Bispec(params, z_min,zmax,spec_int=bs.pk_int,kernels=kernels, simple_kernel = CMB_lens(None,data), k_min=k_min, k_max=k_max, data=data)

        bi_post = PBB.bi_born(ls[0],ls[1],ls[2])

        """ for bias only, general post Born for cross is not yet implemented """
        if cross_bias:
            #pseudo-code
            kernels2= kernels#(gal,cmb,cmb)
            kernels1=(kernels[1],kernels[0],kernels[2])#(cmb,gal,cmb)

            PBB1     = postborn.PostBorn_Bispec(params, z_min,zmax,spec_int=bs.pk_int,kernels=kernels1, simple_kernel = CMB_lens(None,data), k_min=k_min, k_max=k_max, data=data)
            PBB2     = postborn.PostBorn_Bispec(params, z_min,zmax,spec_int=bs.pk_int,kernels=kernels2, simple_kernel = CMB_lens(None,data), k_min=k_min, k_max=k_max, data=data)

            bi_post1 = PBB1.bi_born_cross1(ls[0],ls[1],ls[2])#l1 is associated with galaxy leg
            bi_post2 = PBB2.bi_born_cross2(ls[0],ls[1],ls[2])#l1 is associated with galaxy leg
            #kernel1 is associated with galaxy leg
            bi_post  = bi_post1 + bi_post2
            np.save(bs.filename+"_post_born1.npy",bi_post1)
            np.save(bs.filename+"_post_born2.npy",bi_post2)

        np.save(bs.filename+"_post_born.npy",bi_post)
        np.save(bs.filename+"_post_born_sum.npy",bi_post+bs.bi_phi)

        print(bs.filename+"_post_born.npy")





