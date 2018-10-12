# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:09:11 2016

@author: vboehm
adapted from https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb
"""
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import Cosmology as C
import pickle
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline
from classy import Class

import copy

import kernels


class PostBorn_Bispec():

    def __init__(self,data, zmin, zmax, first_kernel, second_kernel, simple_kernel, k_min=1e-4, k_max=10, lmax=None, acc=4):

        self.kmax  = k_max
        self.kmin  = k_min
        self.zmin  = zmin
        self.zmax  = zmax #this is z_cmb always. For galaxy lensing there is no support in the kernel for chi>chi_source

        # cosmology
        self.cosmo = copy.deepcopy(data.class_params)

        if data is None:
          data     = C.CosmoData(self.cosmo,z=1./np.linspace((1.+zmin)**(-1),(1.+zmax)**(-1),100*acc)-1.)
        self.data  = data

        # lensing and galaxy kernels
        self.first_kernel   = first_kernel
        self.second_kernel  = second_kernel
        self.simple_kernel  = simple_kernel



        self.compute_spec_int()

        if lmax==None:
            lmax=20000
        if acc==None:
            acc=1 #(change back to 1 unless you need high accuracy - much faster)
        self.acc    = acc
        self.nz = 200*acc

        self.ls = np.hstack((np.logspace(-3,0,endpoint=False),np.arange(1, 400, 1),np.arange(400, 2600, 10//self.acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)


        self.lmax   = lmax


        self.chimax = self.data.chi(self.zmax)
        self.chimin = self.data.chi(self.zmin)

        nchimax     = 100*acc
        chimaxs     = np.linspace(self.chimin, self.chimax, nchimax)

        cls = np.zeros((nchimax,self.ls.size))
        for i, chimax in enumerate(chimaxs[1:]):
            cl = self.cl_kappa(chimax)
            cls[i+1,:] = cl
        cls[0,:]=0
        cl_chi_chistar = RectBivariateSpline(chimaxs,self.ls,cls)
        self.inner_int = cl_chi_chistar

        #Get M_*(l,l') matrix
        chis    = np.linspace(self.chimin,self.chimax,self.nz, dtype=np.float64)
        zs      = self.data.zchi(chis)
        dchis   = (chis[2:]-chis[:-2])/2.
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = self.first_kernel(chis, zs)*self.second_kernel(chis,zs)/chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)
        cchi    = cl_chi_chistar(chis,self.ls, grid=True)

        Mstar = np.zeros((self.ls.size,self.ls.size))
        for i, l in enumerate(self.ls):
            k=l/chis
            w[:]=1
            w[k>=self.kmax]=0
            w[k<=self.kmin]=0
            cl = np.dot(dchis*w*self.pk_mm(k,np.log(zs),grid=False)*win,cchi)
            Mstar[i,:] = cl

        self.Mstarsp = RectBivariateSpline(self.ls,self.ls,Mstar)

        self.cl_born(self.chimax)

#    def get_spec_int(self):
#        # load (cross) power spectrum here
#        self.pk_mm = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_))
#
#        self.pk_mh_1st = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_cross_1st))
#
#        self.pk_mh_2nd = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_cross_2nd))



    def compute_spec_int(self, nl=True):

        try:
            self.pk_mm  = pickle.load(open('/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/power_spectra/Pks_pB_tests.pkl','w'))
        except:
            self.cosmo['output'] = 'mPk'
            self.cosmo['P_k_max_1/Mpc'] = self.kmax+1
            self.cosmo['z_max_pk']      = max(self.zmax,1.5)
            self.cosmo.update(C.acc_1)

            if nl:
                self.cosmo['non linear'] = "halofit"
            else:
                self.cosmo['non linear'] = ""

            self.closmo=Class()
            self.closmo.set(self.cosmo)
            print "Initializing CLASS with halofit..."
            print self.cosmo
            self.closmo.compute()
            print 'sigma8 ', self.closmo.sigma8()

            print('zmax',self.zmax)
            a  = np.linspace((1.+self.zmin)**(-1),(1.+self.zmax)**(-1),200)
            z_ = 1/a-1.
            #to prevent fuzzy high ks in interpolated power spectrum at high k
            z2 = np.linspace(self.zmin,self.zmax,200)
            z_ = np.append(z_,z2)
            z_ = np.unique(np.sort(z_))
            a  = (1+z_)**-1


            k_ = np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),200))
            #print(max(k_),self.kmax)
            spec_=np.zeros((len(z_),len(k_)))
            cosmo_pk = self.closmo.pk
            for jj in xrange(len(z_)):
                spec_[jj] = np.asarray([cosmo_pk(kk,z_[jj]) for kk in k_])

            pk_int = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_))

            self.pk_mm = pk_int
            self.pk_mh_1st = pk_int
            self.pk_mh_2nd = pk_int

            pickle.dump(self.pk_mm,open('/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/power_spectra/Pks_pB_tests.pkl','w'))


    def cl_kappa(self, chimax, chimax2=None):

        if chimax2 == None:
          chimax2=chimax

        chis    = np.linspace(self.chimin,chimax,self.nz, dtype=np.float64)
        zs      = self.data.zchi(chis)
        dchis   = (chis[2:]-chis[:-2])/2.
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = self.simple_kernel(chis,zs,chimax)*self.simple_kernel(chis,zs,chimax2)/chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)

        for i, l in enumerate(self.ls):
            k = l/chis
            w[:]=1
            w[k<self.kmin]=0.
            w[k>=self.kmax]=0.
            cl[i] = np.dot(dchis,w*self.pk_mm(k,np.log(zs),grid=False)*win)
        return cl



    def cl_born(self, chimax, chimax2=None):

        chis    = np.linspace(self.chimin,chimax,self.nz, dtype=np.float64)
        zs      = self.data.zchi(chis)
        dchis   = (chis[2:]-chis[:-2])/2.
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = self.first_kernel(chis,zs,chimax)*self.second_kernel(chis,zs,chimax2)/chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)

        for i, l in enumerate(self.ls):
            k =(l+0.5)/chis
            w[:]=1
            w[k<self.kmin]=0.
            w[k>=self.kmax]=0.
            cl[i] = np.dot(dchis,w*self.pk_mm(k,np.log(zs),grid=False)*win)

        self.CL_born=cl



if __name__ == "__main__":

  zmin = 1e-5
  zmax = 1090.

  cosmo = C.Planck2015

  a     = np.linspace(1./(1.+zmin),1./(1.+zmax),100)
  z     = 1./a-1.

  data      = C.CosmoData(cosmo[1],z)

  LSST_bin  = 'all'

  first_kernel  = kernels.CMB_lens(data.chi_cmb,data)
  simple_kernel = kernels.CMB_lens(None,data)
  second_kernel = kernels.CMB_lens(data.chi_cmb,data)#kernels.gal_clus(kernels.dNdz_LSST,kernels.simple_bias,data,LSST_bin)

  PB = PostBorn_Bispec(data, zmin, data.z_cmb, first_kernel, second_kernel, simple_kernel, k_min=1e-4,k_max=10, lmax=None, acc=4)

  L   = np.logspace(1,4,100)
  L1  = np.arange(1,4000)


  phi = np.arange(0,2*np.pi,100)

  res=[]
  for L_ in L:
    res+=[simps(PB.Mstarsp(L_,L1,grid=False)/L1,L1)]

  res= np.asarray(res)
  res*=-1/(2*np.pi)*L**2






