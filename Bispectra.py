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

import HelperFunctions as hf
from scipy.integrate import simps
from scipy.interpolate import splev
import matplotlib.pyplot as plt
from classy import Class
import copy
import time

from scipy.interpolate import RectBivariateSpline


class Bispectra():
    def __init__(self,cosmo,data,l1,l2,l3,z,chi,kernel,config,ang12,ang23,ang31,path,nonlin=False, B_fit=False,k_min=None,k_max=None,fit_z_max=5., ft='SC'):


        self.cosmo      = copy.deepcopy(cosmo)

        self.data       = data
        #for comoving angular diameter distance
        self.z          = z
        self.chi        = chi

        assert((data.z==self.z).all())

        self.l1       = l1
        self.l2       = l2
        self.l3       = l3

        self.L_min    = min(self.l1)
        self.L_max    = max(self.l1)
        self.l_min    = min(self.l3)
        self.l_max    = max(self.l3)

        self.len_bi   = len(self.l1)
        assert(self.len_bi==len(self.l2))

        print "bispectrum size: ", self.len_bi

        self.config   = config

        self.kernel   = kernel

        self.ang12    = ang12
        self.ang23    = ang23
        self.ang31    = ang31

        self.nl         = nonlin

        if self.nl:
            print "Using non-linear matter power spectrum"

        self.ft = ft
        self.B_fit = B_fit
        self.fit_z_max = fit_z_max

        if self.B_fit:
            print "using fitting formula", self.ft
        else:
          self.ft=''

        self.path   = path
        self.kmin   = k_min
        self.kmax   = k_max



    def get_kbounds(self):

        kmax        = max(self.l2)/min(self.chi)
        kmin        = min(self.l2)/max(self.chi)

        if self.kmin==None:
            self.kmin=kmin
        if self.kmax==None:
            self.kmax=kmax

        print "kmin and kmax from ell/chi", kmin, kmax

        self.kmax = min(kmax,self.kmax)
        self.kmin = max(kmin,self.kmin)

        print "kmin and kmax used in calculation", self.kmin, self.kmax


    def __call__(self):
        """
        call method of bispectrum class
        computes the lensing bispectrum
        """

        self.set_up()
        #self.compute_bispectrum_delta()
        #self.compute_bispectrum(kernel1=self.kernel[0], kernel2=self.kernel[1], kernel3=self.kernel[2])
        self.compute_power_spectrum(kernel1=self.kernel[0],kernel2=self.kernel[1])

        self.filename   = self.path+"bispectra/bispec_%s_Lmin%d-Lmax%d-lmax%d_%s_%s"%(self.config,self.L_min,self.L_max,self.l_max,self.cosmo['non linear'],self.ft)

        self.filenameCL   = self.path+"power_spectra/CL_%s_Lmin%d-Lmax%d_%s"%(self.config,self.L_min,self.L_max,self.cosmo['non linear'])

        #np.save(self.filename+'.npy',self.bi_phi)
        np.save(self.filenameCL+'.npy',self.CL)


        try:
            self.closmo.struct_cleanup()
            self.closmo.empty()
        except:
            pass



    def set_up(self):
        """
        initializes all indice-related arrays and instance of class
        """

        self.get_kbounds()

        if self.B_fit:
            k4n=np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),300))
            self.data.get_abc(k4n,self.z[np.where(self.z<=self.fit_z_max)],self.fit_z_max,fit_type=self.ft)

        self.cosmo['output']='tCl, mPk'
        self.cosmo['P_k_max_1/Mpc']= self.kmax
        self.cosmo['z_max_pk']     = max(max(self.z),1.5)
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

        if self.B_fit:
            self.bi_delta_func    = self.bispectrum_delta_fit
        else:
            self.bi_delta_func    = self.bispectrum_delta


        a  = np.linspace((1+min(self.z))**(-1),(1+max(self.z))**(-1),100)
        z_ = 1/a-1.

        plt.figure()
        plt.semilogx(z_,a, ls='', marker='+')
        plt.xlabel('z')
        plt.ylabel('a')
        plt.show()

        plt.figure()
        plt.semilogx(a,self.data.chi(z_), ls='', marker='+')
        plt.xlabel('a')
        plt.ylabel('$\chi$')
        plt.show()

        k_ = np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),80))
        spec_=np.zeros((len(z_),len(k_)))
        cosmo_pk = self.closmo.pk
        for jj in xrange(len(z_)):
            spec_[jj] = np.asarray([cosmo_pk(kk,z_[jj]) for kk in k_])

        self.pk_int = RectBivariateSpline(k_,np.log(z_),np.transpose(spec_))


        #test plots, keep in for now
        z2=np.linspace(min(z_),20.,len(z_))
        plt.figure()
        for z in [0.5, 1.,2.,3.,5.,10.]:
          plt.loglog(k_,self.pk_int(k_,np.log(z),grid=False),marker='+',ls='',markersize=3,label='z=%.1f'%z)
        plt.show()

        plt.figure()
        for jj in np.arange(0,len(z_),3):
          plt.loglog(k_,self.pk_int(k_,np.log(z2[jj]),grid=False),marker='+',ls='',markersize=2, label='z_=%.1f'%z2[jj])
        plt.show()

        plt.figure()
        for jj in np.arange(0,len(z_)):
          plt.loglog(k_,spec_[jj],label='z=%.1f'%z_[jj])
        plt.show()



    def compute_bispectrum_delta(self):
        """
        computes the bispectrum for each chi bin
        -> only use after self.set_up() has been called!
        """

        bi_delta  = np.zeros((len(self.z),self.len_bi))

        beg = time.time()

        for ii in np.arange(0,len(self.z)):
            z_i     = self.z[ii]
            print 'progress in percent ', ii/len(self.z)*100.
            print 'time in min', (time.time()-beg)/60.
            print 'z: ', z_i


            spec1   =[]
            spec2   =[]
            spec3   =[]
            k1      = (self.l1+0.5)/self.chi[ii]
            k2      = (self.l2+0.5)/self.chi[ii]
            k3      = (self.l3+0.5)/self.chi[ii]

            index = np.all([k1>=self.kmin,k2>=self.kmin,k3>=self.kmin,k1<=self.kmax,k2<=self.kmax,k3<=self.kmax],axis=0)

            k1      = k1[index]
            k2      = k2[index]
            k3      = k3[index]
            ang12   = self.ang12[index]
            ang23   = self.ang23[index]
            ang31   = self.ang31[index]


            spec1=self.pk_int(k1,np.log(z_i),grid=False)
            spec2=self.pk_int(k2,np.log(z_i),grid=False)
            spec3=self.pk_int(k3,np.log(z_i),grid=False)

            specs = [spec1,spec2,spec3]

            bi_delta_chi    = self.bi_delta_func(specs, k1, k2, k3, ang12, ang23, ang31, ii)


            bi_delta[ii][index] = bi_delta_chi

        self.bi_delta=np.transpose(bi_delta) #row is now a function of chi




    def bispectrum_delta(self,spectra,k1,k2,k3, ang12, ang23, ang31,ii=None):
        """ returns the bispectrum of the fractional overdensity today (a=1) i.e. B^0, the lowest order in non-lin PT
        *spectrum:   power spectrum for all ks in k_aux
        *k_spec:     array of ks where for which power spectrum is passed
        *k:          array of k's that form the triangles for which the bispectrum is computed
        """
        B =2.*hf.get_F2_kernel(k1,k2,ang12)*spectra[0]*spectra[1]
        B+=2.*hf.get_F2_kernel(k2,k3,ang23)*spectra[1]*spectra[2]
        B+=2.*hf.get_F2_kernel(k3,k1,ang31)*spectra[2]*spectra[0]

        return B


    def bispectrum_delta_fit(self,spectra,k1,k2,k3, ang12, ang23, ang31,i):
        """ returns the bispectrum of the fractional overdensity today (a=1) i.e. B^0, the lowest order in non-lin PT
        *spectrum:   power spectrum for all ks in k_aux
        *k_spec:      array of ks where for which power spectrum is passed
        *k:          array of k's that form the triangles for which the bispectrum is computed
        """
        if self.z[i]<self.fit_z_max:
          B= 2.*self.get_F2_kernel_fit(k1,k2,ang12,i)*spectra[0]*spectra[1]
          B+=2.*self.get_F2_kernel_fit(k2,k3,ang23,i)*spectra[1]*spectra[2]
          B+=2.*self.get_F2_kernel_fit(k1,k3,ang31,i)*spectra[0]*spectra[2]
        else:
          B =2.*hf.get_F2_kernel(k1,k2,ang12)*spectra[0]*spectra[1]
          B+=2.*hf.get_F2_kernel(k2,k3,ang23)*spectra[1]*spectra[2]
          B+=2.*hf.get_F2_kernel(k3,k1,ang31)*spectra[2]*spectra[0]

        return B


    def get_F2_kernel_fit(self,k1,k2,cos,i):

        ak1=splev(k1, self.data.a_nk[i],ext=0)
        ak2=splev(k2, self.data.a_nk[i],ext=0)

        bk1=splev(k1, self.data.b_nk[i],ext=0)
        bk2=splev(k2, self.data.b_nk[i],ext=0)

        ck1=splev(k1, self.data.c_nk[i],ext=0)
        ck2=splev(k2, self.data.c_nk[i],ext=0)

        a=5./7.*ak1*ak2 #growth
        b=0.5*(k1/k2+k2/k1)*cos*bk1*bk2 #shift
        c=2./7.*cos**2*ck1*ck2 #tidal

        F2=a+b+c

        return F2


    def compute_bispectrum(self, kernel1, kernel2=None, kernel3=None):
        """ computes the bispectrum of the lensing potential
        Computes the bispectrum by integration over chi for ever triangle
        """

        kernel1=kernel1(self.chi,self.z)

        if kernel2==None:
            kernel2=kernel1
        else:
          kernel2=kernel2(self.chi,self.z)

        if kernel3==None:
            kernel3=kernel1
        else:
          kernel3=kernel3(self.chi,self.z)

        self.bi_phi = np.zeros(self.len_bi)

        for jj in xrange(self.len_bi):
            integrand       = self.bi_delta[jj]*kernel1*kernel2*kernel3/self.chi**4
            self.bi_phi[jj] = simps(integrand,self.chi)

        #Note: this is without prefactor!
        return True



    def compute_power_spectrum(self, kernel1, kernel2=None):
        """ computes the bispectrum of the lensing potential
        Computes the bispectrum by integration over chi for ever triangle
        """
        L = np.unique(self.l1)
        spec=np.zeros((len(self.z),len(L)))
        for ii in range(len(self.z)):
            k = (np.unique(self.l1))/self.chi[ii]
            index = np.all([k>=self.kmin,k<=self.kmax],axis=0)

            spec[ii][index]=self.pk_int(k[index],np.log(self.z[ii]),grid=False)

        spec = np.transpose(spec)


        kernel1 = kernel1(self.chi,self.z)

        if kernel2==None:
            kernel2=kernel1
        else:
          kernel2=kernel2(self.chi,self.z)


        self.CL = np.zeros(len(L))

        for jj in xrange(len(L)):
            integrand       = spec[jj]*kernel1*kernel2/self.chi**2
            integrand[self.chi==0.]=0.
            self.CL[jj]     = simps(integrand,self.chi)

        return True





