# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:09:11 2016

@author: vboehm
adapted from https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb
"""
from __future__ import division
import sys, os
from matplotlib import pyplot as plt
import numpy as np
import Cosmology as C
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, interp1d
import pickle
import copy
from classy import Class

from Constants import LIGHT_SPEED


class PostBorn_Bispec():

    def __init__(self,cosmo,zmin, zmax, spec_int=None,kernels=(None,None,None), k_min=1e-4,k_max=100, lmax=None, acc=4):

        self.kmax  = k_max
        self.kmin  = k_min

        self.zmin  = zmin
        self.zmax  = zmax

        self.cosmo = copy.deepcopy(cosmo)
        self.data  = C.CosmoData(self.cosmo)

        self.kernel1, self.kernel2, self.kernel3 = kernels

        if lmax==None:
            lmax=20000
        if acc==None:
            acc=4 #(change back to 1 unless you need high accuracy - much faster)

        self.nz = 200*acc

        self.ls = np.hstack((np.arange(2, 400, 1),np.arange(400, 2600, 10//acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)

        self.acc    = acc
        self.lmax   = lmax

        if spec_int is None:
          self.set_spec_int
        else:
          self.spec_int=spec_int

        self.chimax = self.data.chi(self.zmax)
        self.chimin = self.data.chi(self.zmin)

        nchimax     = 100*acc
        chimaxs     = np.linspace(self.chimin, self.chimax, nchimax)

        cls = np.zeros((nchimax,self.ls.size))


        for i, chimax in enumerate(chimaxs[1:]):
            cl = self.cl_kappa(chimax,self.kernel1,self.kernel3)
            cls[i+1,:] = cl
        cls[0,:]=0
        cl_chi_chistar = RectBivariateSpline(chimaxs,self.ls,cls)



        #Get M_*(l,l') matrix
        chis    = np.linspace(self.chimin,self.chimax,self.nz, dtype=np.float64)
        zs      = self.data.z_chi(chis)
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = self.kernel1(chis, zs)*self.kernel2(chis,zs)/chis**2
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
            cl = np.dot(dchis*w*self.spec_int(k,np.log(zs),grid=False)*win,cchi)
            Mstar[i,:] = cl

        self.Mstarsp = RectBivariateSpline(self.ls,self.ls,Mstar)



    def get_spec_int(self):

        self.cosmo['output'] = 'tCl, mPk'
        self.cosmo['P_k_max_1/Mpc'] = self.kmax+1
        self.cosmo['z_max_pk']      = max(max(self.z),1.5)
        if self.nl:
            self.cosmo['non linear'] = "halofit"
        else:
            self.cosmo['non linear'] = ""

        self.closmo=Class()
        self.closmo.set(self.cosmo)
        print "Initializing CLASS with halofit..."
        print self.cosmo
        self.closmo.compute()
        print 'sigma8 ', self.closmo.sigma8()


        a  = np.linspace((1.+self.zmin)**(-1),(1.+max(self.z))**(-1),100)
        z_ = 1/a-1.
        #to prevent fuzzy high ks in interpolated power spectrum at high k
        z2 = np.linspace(min(self.z),max(self.z),100)
        z_ = np.append(z_,z2)
        z_ = np.unique(np.sort(z_))
        a  = (1+z_)**-1


        k_ = np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),80))
        spec_=np.zeros((len(z_),len(k_)))
        cosmo_pk = self.closmo.pk
        for jj in xrange(len(z_)):
            spec_[jj] = np.asarray([cosmo_pk(kk,z_[jj]) for kk in k_])

        self.pk_int = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_))


    def cl_kappa(self, chimax, kernel1, kernel2=None):

        if kernel2 is None:
          kernel2 = kernel1

        chis    = np.linspace(self.chimin,chimax,self.nz, dtype=np.float64)
        zs      = self.data.z_chi(chis)
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = kernel1(chis,chimax)*kernel2(chis,chimax)/chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)

        for i, l in enumerate(self.ls):
            k =(l+0.5)/chis
            w[:]=1
            w[k<self.kmin]=0.
            w[k>=self.kmax]=0.
            cl[i] = np.dot(dchis,w*self.pk_int(k,np.log(zs),grid=False)*win)
        return cl




    def bi_born(self,l1,l2,l3):

        fac= -2./self.data.lens_prefac #check for cross

        cos12 = (l3**2-l1**2-l2**2)/2./l1/l2
        cos23 = (l1**2-l2**2-l3**2)/2./l2/l3
        cos31 = (l2**2-l3**2-l1**2)/2./l3/l1

        res = cos31*cos12/l3/l2*l1**2*(self.Mstarsp(l3,l2,grid=False) +self.Mstarsp(l2,l3,grid=False)) + \
              cos23*cos12/l3/l1*l2**2*(self.Mstarsp(l3,l1,grid=False) +self.Mstarsp(l1,l3,grid=False)) + \
              cos31*cos23/l1/l2*l3**2*(self.Mstarsp(l1,l2,grid=False) +self.Mstarsp(l2,l1,grid=False))

        return res*fac


    def cl_bi_born(self, lset,sym):

        bi   = self.bi_born

        lset = lset.astype(np.float64)
        cl   = np.zeros(lset.shape[0])
        for i, (l1,l2,l3) in enumerate(lset):
            cl[i] = bi(l1,l2,l3)
        return cl





if __name__ == "__main__":
    params=deepcopy(C.SimulationCosmology[1])

