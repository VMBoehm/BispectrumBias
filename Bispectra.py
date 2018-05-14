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
#import matplotlib.pyplot as pl
from classy import Class
import copy
import time



class Bispectra():
    """ Bispectra of
        - delta (function of chi)
        - newtonian potential bi_psi (function of chi)
        - lensing potential bi_phi
    """
    def __init__(self,cosmo,data, ell1,ell2,ell3,z,config,ang12,ang23,ang31, path, z_cmb, b=None, nonlin=False, B_fit=False, kkg=False, kgg=False, kkk=True, dndz=None,norm=None,k_min=None,k_max=None,sym=False, fit_z_max=1.5):


        self.cosmo      = copy.deepcopy(cosmo)

        self.data       = data
        #for comoving angular diameter distance
        self.chi        = self.data.chi(z)
        self.chi_cmb    = self.data.chi(z_cmb)
        print "chi_cmb [Mpc]: %f"%self.chi_cmb

        self.z          = z
        self.z_cmb      = z_cmb

        assert((data.z==self.z).all())

        self.l1       = ell1
        self.l2       = ell2
        self.l3       = ell3

        self.L_min    = min(self.l1)
        self.L_max    = max(self.l1)
        self.l_min    = min(self.l3)
        self.l_max    = max(self.l3)

        self.len_bi   = len(self.l1)
        assert(self.len_bi==len(self.l2))

        print "bispectrum size: ", self.len_bi

        self.config   = config

        self.set_stage= False

        self.ang12    = ang12
        self.ang23    = ang23
        self.ang31    = ang31


        self.kkg        = kkg
        self.kgg        = kgg
        self.kkk        = kkk
        assert(kkk+kgg+kkg==1)

        self.dndz       = dndz
        self.norm       = norm

        if kkg:
            print 'Computing $B_{\phi\phig}$'

        elif kgg:
            print 'Computing $B_{\phi g g}$'
        elif kkk:
            print 'Computing $B_{\phi \phi \phi}$'

        self.nl         = nonlin
        if self.nl:
            print "Using non-linear matter power spectrum"


        self.B_fit     = B_fit
        self.fit_z_max = fit_z_max
        if self.B_fit:
            print "using Gil-Marin et al. fitting formula"

        self.path   = path+'bispectra/'

        self.b      = b

        self.sym    = sym

        self.kmin   = k_min
        self.kmax   = k_max
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

        self.filename   = self.path+"bispec_phi_%s_Lmin%d-Lmax%d-lmax%d-lenBi%d"%(self.config,self.L_min,self.L_max,self.l_max,self.len_bi)
        if self.sym:
            self.filename+='_sym'
        try:
            assert(False)
            self.bi_phi=np.load(self.filename+'.npy')
            print "loading file %s"%(self.filename+'.npy')
        except:
            print "%s not found \n Computing Bispectrum of overdensity..."%self.filename
            self.set_up()
            self.compute_Bispectrum_delta()
            self.compute_Bispectrum_Phi()
            np.save(self.filename+'.npy',self.bi_phi)

        try:
            self.closmo_lin.struct_cleanup()
            self.closmo_lin.empty()
        except:
            pass
        try:
            self.closmo_nl.struct_cleanup()
            self.closmo_nl.empty()
        except:
            pass


    def set_up(self):
        """
        initializes all indice-related arrays and instance of class
        """

#TODO: check!
        if self.B_fit:
            k4n=np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),100))
            k4n=np.concatenate((k4n,np.exp(np.linspace(np.log(1e-3),np.log(0.5),100))))[:-1]
            k4n=np.sort(k4n)
            self.data.get_abc(k4n,self.z[np.where(self.z<=self.fit_z_max)],self.fit_z_max)


        self.cosmo['output']='tCl, mPk'


        self.cosmo['P_k_max_1/Mpc']= self.kmax
        self.cosmo['z_max_pk']     = max(max(self.z),1.5)

        print self.cosmo
        #Initializing class
        if self.nl==False:
            self.closmo_lin=Class()
            self.closmo_lin.set(self.cosmo)
            self.cosmo['non linear'] = ""
            print "Initializing CLASS..."
            print self.cosmo
            self.closmo_lin.compute()
            print 'sigma8 ', self.closmo_lin.sigma8()
        else:
            self.closmo_lin=None

        if self.nl:
            self.cosmo['non linear'] = "halofit"
            self.closmo_nl=Class()
            self.closmo_nl.set(self.cosmo)
            print "Initializing CLASS with halofit..."
            print self.cosmo
            self.closmo_nl.compute()
            print 'sigma8 ', self.closmo_nl.sigma8()
        else:
            self.closmo_nl=None

        self.set_stage=True






    def compute_Bispectrum_delta(self):
        """
        computes the bispectrum for each chi bin
        -> only use after self.set_up() has been called!
        """

        bi_delta  = np.zeros((len(self.z),self.len_bi))

        if self.nl:
            cosmo_pk = self.closmo_nl.pk
        else:
            cosmo_pk = self.closmo_lin.pk

        beg = time.time()
        for ii in np.arange(0,len(self.z)):


            z_i     = self.z[ii]
            print ii/len(self.z)*100.
            print (time.time()-beg)/60., 'min'
            print 'z: ', z_i

            spec1   =[]
            spec2   =[]
            spec3   =[]

            k1      = (self.l1+0.5)/self.chi[ii]
            k2      = (self.l2+0.5)/self.chi[ii]
            k3      = (self.l3+0.5)/self.chi[ii]


            index=np.all([k1>=self.kmin,k2>=self.kmin,k3>=self.kmin,k1<=self.kmax,k2<=self.kmax,k3<=self.kmax],axis=0)

            k1      = k1[index]
            k2      = k2[index]
            k3      = k3[index]
            ang12   = self.ang12[index]
            ang23   = self.ang23[index]
            ang31   = self.ang31[index]

            print len(k1)
            a=time.time()
            for j in xrange(len(k1)):
                spec1+=[cosmo_pk(k1[j],z_i)]
                spec2+=[cosmo_pk(k2[j],z_i)]
                spec3+=[cosmo_pk(k3[j],z_i)]
            print (time.time()-a)/60.



            specs = [np.asarray(spec1),np.asarray(spec2),np.asarray(spec3)]

            if self.B_fit==False:
                    bi_delta_chi    = self.bispectrum_delta(specs,k1,k2,k3,ang12, ang23, ang31)

            elif self.B_fit and self.z[ii]<=self.fit_z_max:
                    bi_delta_chi    = self.bispectrum_delta_fit(specs,k1,k2,k3, ang12, ang23, ang31,ii)
            elif self.B_fit and self.z[ii]>self.fit_z_max:
                    bi_delta_chi    = self.bispectrum_delta(specs,k1,k2,k3, ang12, ang23, ang31)
            else:
                raise ValueError('Something went wrong with matter bispectrum specifications')

            bi_delta[ii][index] = bi_delta_chi

        self.bi_delta=np.transpose(bi_delta) #row is now a function of chi




    def bispectrum_delta(self,spectra,k1,k2,k3, ang12, ang23, ang31):
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

        B= 2.*self.get_F2_kernel_fit(k1,k2,ang12,i)*spectra[0]*spectra[1]
        B+=2.*self.get_F2_kernel_fit(k2,k3,ang23,i)*spectra[1]*spectra[2]
        B+=2.*self.get_F2_kernel_fit(k1,k3,ang31,i)*spectra[0]*spectra[2]

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


    def compute_Bispectrum_Phi(self):
        """ computes the bispectrum of the lensing potential
        Computes the bispectrum by integration over chi for ever triangle
        """
        if self.set_stage==False:
            self.set_up()

        index   = np.arange(self.len_bi)

        W_lens  = ((self.chi_cmb-self.chi)/(self.chi_cmb*self.chi))*(self.z+1.)

        if self.kkg:
            dchidz  = self.data.dchidz(self.z)
            dzdchi  = 1./dchidz
            W_gal   = self.b*self.dndz(self.z)/self.norm*dzdchi
            kernel  = W_gal*W_lens**2
        elif self.kgg:
            dchidz  = self.data.dchidz(self.z)
            dzdchi  = 1./dchidz
            W_gal   = self.b*self.dndz(self.z)/self.norm*dzdchi
            kernel  = W_gal**2*W_lens/self.chi**2
        else:
            kernel  = W_lens**3*self.chi**2


        bi_phi=[]
        for jj in index:
            integrand   = self.bi_delta[jj]*kernel
            bi_phi      +=[simps(integrand,self.chi)]
        self.bi_phi=np.array(bi_phi)

        if self.kkg:
            if self.sym:
                fac = self.data.prefacs**2*(1./(self.l1*self.l2)**2\
                +1./(self.l1*self.l3)**2+1./(self.l2*self.l3)**2)
            else:
                #for bias code, L=associated with galaxy leg
                fac = self.data.prefacs**2*(1./(self.l2*self.l3)**2)
#                #for S/N code, L3 =Ll=associated with galaxy leg
#                else:
#                    fac = self.data.prefacs**2*(1./((self.ell[0::3]+0.5)*(self.ell[2::3]+0.5))**2)
        elif self.kgg:
            if self.sym:
                fac =self.data.prefacs*(1./self.l1**2+1./self.l2**2+1./self.l3**2)
            else:
                fac =self.data.prefacs/(self.l1)**2
        else:
            if self.sym:
                print 'Note: no symmetrization for auto spectrum!'
            fac =(self.data.prefacs**3)/(self.l1*self.l2*self.l3)**2
        self.bi_phi*=fac


        return True









