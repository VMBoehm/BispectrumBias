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
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, interp1d
import copy

import kernels


class PostBorn_Bispec():

    def __init__(self,cosmo, zmin, zmax, lens_kernel, gal_kernel, simple_kernel, k_min=1e-4,k_max=1e-1, lmax=None, acc=4,data=None):

        # k- boundaries
        self.kmax  = k_max
        self.kmin  = k_min

        # redshift boundaries
        self.zmin  = zmin
        self.zmax  = zmax

        # cosmology
        self.cosmo = copy.deepcopy(cosmo)

        if data is None:
          data = C.CosmoData(self.cosmo,z=1./np.linspace((1.+zmin)**(-1),(1.+zmax)**(-1),100)-1.)

        self.data  = data

        # lensing and galaxy kernels
        self.lens_kernel = lens_kernel
        self.gal_kernel  = gal_kernel
        self.simple_kernel = simple_kernel

        if lmax==None:
            lmax=20000
        if acc==None:
            acc=4 #(change back to 1 unless you need high accuracy - much faster)

        self.nz = 200*acc

        self.ls = np.hstack((np.arange(2, 400, 1),np.arange(400, 2600, 10//acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)

        self.acc    = acc
        self.lmax   = lmax


        self.chimax = self.data.chi(self.zmax)
        self.chimin = self.data.chi(self.zmin)

        nchimax     = 100*acc
        chimaxs     = np.linspace(self.chimin, self.chimax, nchimax)

        cls = np.zeros((nchimax,self.ls.size))


        for i, chimax in enumerate(chimaxs[1:]):
            cl = self.cl_kappa(chimax,self.simple_kernel,self.lens_kernel)
            cls[i+1,:] = cl
        cls[0,:]=0
        cl_chi_chistar = RectBivariateSpline(chimaxs,self.ls,cls)


        #Get M_*(l,l') matrix
        chis    = np.linspace(self.chimin,self.chimax,self.nz, dtype=np.float64)
        zs      = self.data.zchi(chis)
        dchis   = (chis[2:]-chis[:-2])/2.
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = self.lens_kernel(chis, zs)*self.gal_kernel(chis,zs)
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)
        cchi    = cl_chi_chistar(chis,self.ls, grid=True)

        Mstar = np.zeros((self.ls.size,self.ls.size))
        for i, l in enumerate(self.ls):
            k=(l+0.5)/chis
            w[:]=1
            #should take care of everything
            w[k>=self.kmax]=0
            w[k<=self.kmin]=0
            cl = np.dot(dchis*w*self.pk_int_cross(k,np.log(zs),grid=False)*win,cchi)
            Mstar[i,:] = cl

        self.Mstarsp = RectBivariateSpline(self.ls,self.ls,Mstar)



    def get_spec_int(self):

        # load (cross) power spectrum here
        self.pk_int = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_))

        self.pk_int_cross = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_cross))


    def cl_kappa(self, chimax, simple_kernel, lens_kernel):


        chis    = np.linspace(self.chimin,chimax,self.nz, dtype=np.float64)
        zs      = self.data.zchi(chis)
        dchis   = (chis[2:]-chis[:-2])/2.
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = lens_kernel(chis,zs,chimax)*simple_kernel(chis,zs)*chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)

        for i, l in enumerate(self.ls):
            k =(l+0.5)/chis
            w[:]=1
            w[k<self.kmin]=0.
            w[k>=self.kmax]=0.
            cl[i] = np.dot(dchis,w*self.pk_int(k,np.log(zs),grid=False)*win)
        return cl





if __name__ == "__main__":

  zmin = 1e-5
  zmax = 5

  cosmo = C.Chirag

  a     = np.linspace(1./(1.+zmin),1./(1.+zmax),100)
  z     = 1./a-1.

  data      = C.CosmoData(cosmo,z)

  LSST_bin  = 'all'

  lens_kernel   = kernels.CMB_lens(data.chicmb,cosmo)
  simple_kernel = lens_kernel
  gal_kernel    = kernels.gal_clus(kernels.dNdz_LSST,1.,data,LSST_bin)

  PB = PostBorn_Bispec(data,zmin, zmax, lens_kernel, gal_kernel, simple_kernel, k_min=1e-4,k_max=1e-1, lmax=None, acc=4,data=data)

  L = np.arange(0,4000)
  L1= np.arange(0,4000)

  res=[]
  for L_ in L:
    res+=[np.simps(PB.Mstarsp(L_,L1,grid=False)/L1,L1)]

  res= np.asarray(res)
  res*=-1/(2*np.pi)





