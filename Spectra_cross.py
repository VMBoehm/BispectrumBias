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
from scipy.interpolate import interp1d, splev
import pickle
from copy import deepcopy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.ioff()

from classy import Class

import Cosmology as C
from N32biasIntegrals import I0, I2#, bi_cum
from BkkgSkewness import beta_RR as skew
import CAMB_postborn as postborn
from Constants import LIGHT_SPEED



class Bispectra():
    """ Bispectra of 
        - delta (function of chi)
        - newtonian potential bi_psi (function of chi)
        - lensing potential bi_phi
    """
    def __init__(self,cosmo,data, ell,z,config,ang12,ang23,ang31, path, z_cmb, b=None, nonlin=False, B_fit=False, kkg=False, kgg=False, dndz=None,norm=None,k_min=None,k_max=None,sym=False, fit_z_max=1.5, cross_bias=False):
        """
        initializes/computes all three bispectra
        * cosmo:    instance of class Cosmology
        * data:     instance of class CosmoData
        * ell:      array of ell vector absolute values
        * z:        array of redshifts
        * config:   string encoding the configuration of this run
        * ang12,23,31: array of angles between vectors
        """
        print "nl", nonlin
        print "\n"

        self.cosmo      = cosmo
        
        self.ell        = ell    
        
        self.data       = data
        #for comoving angular diameter distance
        self.chi        = self.data.chi(z)
        self.chi_cmb    = self.data.chi(z_cmb)
        print "chi_cmb [Mpc]: %f"%self.chi_cmb
        self.z          = z
        self.z_cmb      = z_cmb
        assert((data.z==self.z).all())
  
        self.L_min    = min(ell[0::3])
        self.L_max    = max(ell[0::3])
        self.l_min    = min(ell[2::3])
        self.l_max    = max(ell[2::3])
        
        self.bin_num    = len(ell)
        self.len_bi     = self.bin_num/3
        
        print "L min: ", self.L_min
        print "L max: ", self.L_max
        print "l min: ", self.l_min
        print "l max: ", self.l_max
        
        print "bispectrum size: ", self.len_bi
        
        self.kmin       = k_min
        self.kmax       = k_max
        
        self.config     = config
        print "configuration: ", self.config
        #has set_up been called?
        self.set_stage  = False
        
        self.ang12      = ang12
        self.ang23      = ang23
        self.ang31      = ang31
        

        self.kkg        = kkg
        self.kgg        = kgg
        self.dndz       = dndz
        self.norm       = norm
        
        self.cross_bias = cross_bias
        
        if kkg:
            print 'Computing $B_{\phi\phig}$'
            
        if kgg:
            print 'Computing $B_{\phi g g}$'
        
        self.nl         = nonlin
        if self.nl:
            print "Using non-linear matter power spectrum"	


        self.B_fit  = B_fit
        self.fit_z_max = fit_z_max
        if self.B_fit:
            print "using Gil-Marin et al. fitting formula"
        
        self.path   = path+'cross_bias_spectra/'
        
        self.b      = b
        
        self.sym    = sym
        
        
        
    def __call__(self):
        """
        call method of bispectrum class
        computes the lensing bispectrum
        """

        self.filename=self.path+"bispec_phi_%s_Lmin%d-Lmax%d-lmax%d-lenBi%d"%(self.config,self.L_min,self.L_max,self.l_max,self.len_bi)
        if self.sym:
            self.filename+='_sym'
        if self.cross_bias:
            self.filename+='_4bias'
        try:
            self.bi_phi=np.load(self.filename+'.npy')
            print "loading file %s"%(self.filename+'.npy')									
        except:
            print "%s not found \n Computing Bispectrum of overdensity..."%self.filename
            self.filename2=self.path+"bispec_delta_%s_Lmin%d-Lmax%d-lmax%d-lenBi%d"%(self.config,self.L_min,self.L_max,self.l_max,self.len_bi)
            try:
                self.bi_delta=np.load(self.filename2+'.npy')
            except:
                print "%s not found \n Computing Bispectrum of Newtonian Potential..."%(self.filename2+'.npy')
                self.compute_Bispectrum_delta()
                np.save(self.filename2+'.npy',self.bi_delta)
            
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
       
       
        #for Limber matter power spectrum 
        self.pow_ell     = np.linspace(min(self.ell)/10.,max(self.ell)+1.,1000)
        #number of unique l values
        self.powspec_len = np.int(len(self.pow_ell))

        #k=ell/chi, ndarray with ell along rows, chi along columns
        self.k  = np.outer(1./self.chi,self.ell+0.5)
	 
        		
        kmax    = max(self.pow_ell)/min(self.chi)
        kmin    = min(self.pow_ell)/max(self.chi)
        print k_min, min(self.pow_ell)/min(self.chi)
        if self.kmin==None:
            self.kmin=kmin
        if self.kmax==None:
            self.kmax=kmax        				
        print "kmin and kmax from ell/chi", self.kmin,self.kmax

        if self.B_fit:
            k4n=np.exp(np.linspace(np.log(self.kmin),np.log(self.kmax),100))
            k4n=np.concatenate((k4n,np.exp(np.linspace(-4,-1,100))))
            k4n=np.sort(k4n)
            self.data.get_abc(k4n,self.z[np.where(self.z<=self.fit_z_max)],fit_z_max)


        print "kmin and kmax for bispectrum calculation", self.kmin,self.kmax 
        if self.cosmo.class_params['output']!='tCl, lCl, mPk':
            if self.L_max<=3000:
                self.cosmo.class_params['output']='tCl, pCl, lCl, mPk'
                print 'Calculating lensed Pol spectra'
            else:
                self.cosmo.class_params['output']='tCl, lCl, mPk'
            #self.cosmo.class_params['lensing']='yes'
            self.cosmo.class_params['tol_perturb_integration']=1.e-6
					
        self.cosmo.class_params['l_max_scalars']=min(self.L_max,4000)+2000
        self.cosmo.class_params['z_max_pk'] = max(z)	
        print max(z), self.z_cmb
        #Maximum k value in matter power spectrum
        self.cosmo.class_params['P_k_max_1/Mpc'] = self.kmax
        self.cosmo.class_params['k_min_tau0'] = self.kmin*13000.


        #Initializing class
        if self.nl==False:
            self.closmo_lin=Class()
            self.closmo_lin.set(self.cosmo.class_params)
            self.cosmo.class_params['non linear'] = ""
            print "Initializing CLASS..."
            print self.cosmo.class_params
            self.closmo_lin.compute()
            cl_unl=self.closmo_lin.raw_cl(min(self.L_max,4000))
            try:
                cl_len=self.closmo_lin.lensed_cl(min(self.L_max,4000))
            except:
                pass
            print self.closmo_lin.get_current_derived_parameters(['sigma8'])
        else:
            self.closmo_lin=None

        if self.nl:
            self.cosmo.tag+='_nl'
            self.cosmo.class_params['non linear'] = "halofit"
            self.closmo_nl=Class()
            self.closmo_nl.set(self.cosmo.class_params)
            print "Initializing CLASS with halofit..."
            print self.cosmo.class_params
            self.closmo_nl.compute()
            print self.closmo_nl.get_current_derived_parameters(['sigma8'])
            cl_unl=self.closmo_nl.raw_cl(min(self.L_max,4000))
            try:
                cl_len=self.closmo_nl.lensed_cl(min(self.L_max,4000))
            except:
                pass
        else:
            self.closmo_nl=None
            
        self.set_stage=True
        try:
            print self.cosmo.class_params['A_s']
        except:
            print self.cosmo.class_params['ln10^{10}A_s']
        
        try:
            pickle.dump([self.cosmo.class_params,cl_unl,cl_len],open('../class_outputs/class_cls_%s_lensed.pkl'%self.cosmo.tag,'w'))
        except:
            pickle.dump([self.cosmo.class_params,cl_unl],open('../class_outputs/class_cls_%s_le.pkl'%self.cosmo.tag,'w'))
            
        if equilat:
            pickle.dump([self.chi,self.z,self.k],open(self.path+'z_k_equilat.pkl','w'))
            
            
#        h   = self.cosmo.class_params['h']
#        k_  = np.exp(np.linspace(np.log(1e-4*h),np.log(100.*h),100))
#        pl.figure()
#        for z_ in [0.,1.,10.]:
#            plk =[]
#            for kk in k_:
#                plk+=[self.closmo_nl.pk(kk,z_)]
#            pl.loglog(k_/h,np.asarray(plk)*h**3,label='z=%d'%z_)
#            print min(k_/h), max(k_/h)
#        pl.xlim(1e-4,1.) 
#        pl.ylim(0.1,100000)
#        pl.xlabel(r'$k [h/Mpc]$')
#        pl.ylabel(r'$P(k) [Mpc/h]^3$')
#        pl.legend(loc='best')
#        pl.savefig('pow_spec_nl.png')
        
        
            
        
        
            
    def compute_Bispectrum_delta(self):
        """ 
        computes the bispectrum for each chi bin
        -> only use after self.set_up() has been called! 
        """      
        if self.set_stage==False:
            self.set_up()
        
        bi_delta  = np.ndarray((len(self.z),self.len_bi))
     
        #more exact Limber, k for power spectrum
        #ell     = np.sqrt(self.pow_ell*(self.pow_ell+np.ones(len(self.pow_ell))))
        
        if self.nl:
            cosmo_pk = self.closmo_nl.pk
        else:
            cosmo_pk = self.closmo_lin.pk
        for i in np.arange(0,len(self.z)):
            
            print i
            
            z_i    = self.z[i]
            k_i    = self.k[i]
            
            k_spec = (self.pow_ell/self.chi[i])
            k_spec = k_spec[np.where((k_spec>self.kmin)*(k_spec<self.kmax))]
            spec=[]

            for j in np.arange(0,len(k_spec)):
                spec+=[cosmo_pk(k_spec[j],z_i)]
            spec=np.array(spec)
            
            if self.B_fit==False:
                    bi_delta_chi    = self.bispectrum_delta(spec,k_spec,k_i)
                    
            elif self.B_fit and self.z[i]<=self.fit_z_max:
                    bi_delta_chi    = self.bispectrum_delta_fit(spec,k_spec,k_i,i)
            elif self.B_fit and self.z[i]>self.fit_z_max:
                    bi_delta_chi    = self.bispectrum_delta(spec,k_spec,k_i)
            else:
                raise ValueError('Something went wrong with matter bispectrum specifications')
																				
            bi_delta[i] = bi_delta_chi#*self.delta2psi(k_i,len(bi_delta_chi),i)
                   
        self.bi_delta=np.transpose(bi_delta) #row is now a function of chi
        del self.k
        
        

    def bispectrum_delta(self,spectrum,k_spec,k):
        """ returns the bispectrum of the fractional overdensity today (a=1) i.e. B^0, the lowest order in non-lin PT
        *spectrum:   power spectrum for all ks in k_aux
        *k_spec:     array of ks where for which power spectrum is passed
        *k:          array of k's that form the triangles for which the bispectrum is computed
        """
        spec   = interp1d(np.log(k_spec),np.log(spectrum),kind="slinear",bounds_error=False, fill_value=0.)
    
        k1       = k[::3]
        k2       = k[1::3]
        k3       = k[2::3]

        B=2.*hf.get_F2_kernel(k1,k2,self.ang12)*np.exp(spec(np.log(k1)))*np.exp(spec(np.log(k2)))
        B+=2.*hf.get_F2_kernel(k2,k3,self.ang23)*np.exp(spec(np.log(k2)))*np.exp(spec(np.log(k3)))
        B+=2.*hf.get_F2_kernel(k1,k3,self.ang31)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
        index   =np.where(np.any([(k1>self.kmax),(k1<self.kmin)],axis=0))
        B[index]=0.
        index   =np.where(np.any([(k2>self.kmax),(k2<self.kmin)],axis=0))
        B[index]=0.
        index   =np.where(np.any([(k3>self.kmax),(k3<self.kmin)],axis=0))
        B[index]=0.
        						
        return B 

                                     
    def bispectrum_delta_fit(self,spectrum,k_spec,k,i):
        """ returns the bispectrum of the fractional overdensity today (a=1) i.e. B^0, the lowest order in non-lin PT
        *spectrum:   power spectrum for all ks in k_aux
        *k_spec:      array of ks where for which power spectrum is passed
        *k:          array of k's that form the triangles for which the bispectrum is computed
        """
        spec  = interp1d(np.log(k_spec),np.log(spectrum),kind="slinear",bounds_error=False, fill_value=0.)
		
        k1       = k[::3]
        k2       = k[1::3]
        k3       = k[2::3]

        B= 2.*self.get_F2_kernel_fit(k1,k2,self.ang12,i)*np.exp(spec(np.log(k1)))*np.exp(spec(np.log(k2)))
        B+=2.*self.get_F2_kernel_fit(k2,k3,self.ang23,i)*np.exp(spec(np.log(k2)))*np.exp(spec(np.log(k3)))
        B+=2.*self.get_F2_kernel_fit(k1,k3,self.ang31,i)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
        index=np.where(np.any([(k1>self.kmax),(k1<self.kmin)],axis=0))
        B[index]=0.
        index=np.where(np.any([(k2>self.kmax),(k2<self.kmin)],axis=0))
        B[index]=0.
        index=np.where(np.any([(k3>self.kmax),(k3<self.kmin)],axis=0))
        B[index]=0.

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
        dchidz  = self.data.dchidz(self.z)
        dzdchi  = 1./dchidz
        if self.kkg:
            W_gal   = self.b*self.dndz(self.z)/self.norm*dzdchi 
            kernel  = W_gal*W_lens**2
        elif self.kgg:
            W_gal   = self.b*self.dndz(self.z)/self.norm*dzdchi
            kernel  = W_gal**2*W_lens/self.chi**2      
        else:
            kernel  = W_lens**3*self.chi**2
            
        pl.figure()
        pl.loglog(self.z,W_gal*W_lens**2,label='ppg')
        pl.loglog(self.z,W_lens**3*self.chi**2*self.data.prefacs,label='ppp*fac')
        pl.legend()
        pl.gca().set_ylim(bottom=1e-15)
        pl.savefig('Kernel_Comparison.png')

        bi_phi=[]
        for j in index:
            integrand   = self.bi_delta[j]*kernel
            bi_phi+=[simps(integrand,self.chi)]
        self.bi_phi=np.array(bi_phi)

        
        if self.kkg:
            if self.sym:
                fac = self.data.prefacs**2*(1./((self.ell[1::3]+0.5)*(self.ell[2::3]+0.5))**2\
                +(1./((self.ell[1::3]+0.5)*(self.ell[0::3]+0.5)))**2\
                +(1./((self.ell[2::3]+0.5)*(self.ell[0::3]+0.5)))**2)
            else:
                #for bias code, L=associated with galaxy leg
                if self.cross_bias:
                    fac = self.data.prefacs**2*(1./((self.ell[1::3]+0.5)*(self.ell[2::3]+0.5))**2)
                #for S/N code, L3 =Ll=associated with galaxy leg
                else:
                    fac = self.data.prefacs**2*(1./((self.ell[0::3]+0.5)*(self.ell[2::3]+0.5))**2)
        elif self.kgg:
            if self.sym:
                fac =self.data.prefacs*(1./(self.ell[0::3]+0.5)**2+1./(self.ell[1::3]+0.5)**2+1./(self.ell[2::3]+0.5)**2)
            else:
                fac =self.data.prefacs/(self.ell[0::3]+0.5)**2
        else:
            fac =(self.data.prefacs)**3/((self.ell[1::3]+0.5)*(self.ell[2::3]+0.5)*(self.ell[0::3]+0.5))**2
        self.bi_phi*=fac
        return True
        



"""---- Choose you settings here ---"""
if __name__ == "__main__":  
    
    "---begin settings---"
    kkg         = True
    kgg         = False
    cross_bias  = True
    
    LSST        = True
    
    sym         = False

    equilat     = False
    folded      = False
    
    integrals   = True
    
    tag         = ''
    
    assert(kkg+kgg<=1)
    
    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = True    
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662
    post_born   = True
    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
    
    # smoothing scale for integrated bispectrum
    rad         = np.linspace(10,500)

    #binbounds
    bounds      = {'0':[0.0,0.5],'1':[0.5,1.],'2':[1.,2.]}

    #number of redshift bins 
    bin_num     = 200
    
    #sampling in L/l and angle
    len_L       = 140
    len_l       = 140
    len_ang     = 220

    #ell range (for L and l)
    L_min       = 1.
    L_max       = 10000.
    
    l_min       = 1.
    l_max       = 10000.
    
    k_min       = 1e-4
    k_max       = 100.
    
    fit_z_max   = 1.5
    
    #tag for L-sampling
    ell_type    ="linlog_halfang"
    
    if equilat:
        ell_type="equilat"
        len_side= 250
    if folded:
        ell_type="folded"
        len_side= 250
    
    Delta_theta = 1e-4
    
    nl          = True
    
    cparams     = C.Planck2015_TTlowPlensing        
				    

    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'
    #path, where to store results
    path            = "/afs/mpa/temp/vboehm/spectra/"
    
    "---end settings---"

    for red_bin in ['0','1','2','None']:
        
        if LSST: 
            dn_filename = 'dndz_LSST_i27_SN5_3y'
     
        #initialize cosmology
        params  = deepcopy(cparams)
        cosmo   = C.Cosmology(zmin=0.00, zmax=1190, Params=params, Limber=Limber, mPk=False, Neutrinos=False,lensing=True)
        closmo  = Class()
        closmo.set(cosmo.class_params)
        closmo.compute()
        #set up z range and binning in z space
        z_min   = 1e-4
        z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
        closmo.struct_cleanup()
        closmo.empty()
        
        print "z_cmb: %f"%z_cmb
    
        z       = np.exp(np.linspace(np.log(z_min),np.log(z_cmb-0.01),bin_num))
        
        if kkg or kgg:
            if LSST:
                gz, dgn = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
                dndz    = interp1d(gz, dgn, kind='linear')
                z_g     = np.linspace(max(bounds[red_bin][0],z_min),bounds[red_bin][1],bin_num)
                dndz    = dndz(z_g)
                dndz    = interp1d(z_g, dndz, kind='linear',bounds_error=False,fill_value=0.)
                bias    = z+1.
            else:
                z0      = 1./3.
                dndz    = (z/z0)**2*np.exp(-z/z0)
                dndz    = interp1d(z,dndz,kind='slinear',fill_value=0.,bounds_error=False)
                bias    = z+1.
                
            norm    = simps(dndz(z),z)
        
            print 'norm ', norm
            
        else:
            dndz    = None
            norm    = None
            bias    = None
        #cosmo dependent functions
        data    = C.CosmoData(cosmo,z)
        print 'z:', z
        #list of all triangles and their sides (loading is faster than recomputing)
        filename=path+"ell_%s_Lmin%d_Lmax%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_max,len_L,len_l,len_ang,Delta_theta)
        filename_ang=path+"ang_%s_Lmin%d_Lmax%d_lmax%d_lenL%d_lenl%d_lenang%d_%.0e.pkl"%(ell_type,L_min,L_max,l_max,len_L,len_l,len_ang,Delta_theta)
        
        print filename
            
        try:
            ell=pickle.load(open(filename))
            angles=pickle.load(open(filename_ang))
            ang12=angles[0]
            ang23=angles[1]
            ang31=angles[2]
            angmu=angles[3]
        except:
            ell         = []
            print "ell file not found"
            if ell_type=="linlog_halfang":
                #L = |-L|, equally spaced in lin at low L and in log at high L 
                La        = np.arange(L_min,50)
                Lb        = np.exp(np.linspace(np.log(50),np.log(L_max),len_L-49))
                L         = np.append(La,Lb)
                #l 
                la        = np.arange(l_min,50)
                lb        = np.exp(np.linspace(np.log(50),np.log(l_max),len_l-49))
                l         = np.append(la,lb)     
                
            elif ell_type=='special_halfang':
                acc       = 2
                L         = np.hstack((np.arange(1, 20, 2), np.arange(25, 200, 10//acc), np.arange(220, 1200, 30//acc),np.arange(1200, min(10000,2600), 150//acc),np.arange(2600, 10000+1, 1000//acc)))
                l         = L
                len_L     = len(L)
                len_l     = len(l)

            elif ell_type=="lin_halfang":
                #L = |-L|, equally spaced in lin
                L         = np.linspace(L_min,L_max,len_L)
                l         = np.linspace(l_min,l_max,len_l)
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
            else:
                raise Exception("ell type not consistent with any sampling method")
                
            # angle, cut edges to avoid numerical instabilities
            #TODO: try halving this angle, probably requires multiplication by 2, but should avoid l2=0
            theta   = np.linspace(Delta_theta,np.pi-Delta_theta, len_ang)
            if ell_type=='equilat':
                theta   = np.asarray([np.pi/3.]*len_side)
            if ell_type=='folded':
                theta   = np.asarray([0.]*len_side)
                
            cosmu   = np.cos(theta) #Ldotl/Ll or -l1dotl3/l1/l3 (l1+l2+l3=0) (angle used in beta Integrals)
            
            ang31=[]
            ang12=[]
            ang23=[]
            angmu=[]
    
            sqrt=np.sqrt
            #all combinations of the two sides and the angles
            
            if equilat:
                for i in range(len_side):
                        l1= L[i]
                        l3= L[i]
                        l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
                        if l2<1e-5:
                            l2=1e-5
                        ell+=[l1]+[l2]+[l3]
                        ang31+=[-cosmu[i]]
                        ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                        ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                        angmu+=[theta[i]]
            elif folded:
                for i in range(len_side):
                        l1= L[i]
                        l3= l[i]
                        l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[i])
                        if l2<1e-5:
                            l2=1e-5
                        ell+=[l1]+[l2]+[l3]
                        ang31+=[-cosmu[i]]
                        ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                        ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                        angmu+=[theta[i]]                
            else:                       
                for i in range(len_L):
                    for j in range(len_l):
                        for k in range(len_ang):
                            l1= L[i]
                            l3= l[j]
                            l2= sqrt(l1*l1+l3*l3-2.*l1*l3*cosmu[k])
                            if l2<1e-5:
                                l2=1e-5
                            ell+=[l1]+[l2]+[l3]
                            ang31+=[-cosmu[k]]
                            ang12+=[(l3*l3-l1*l1-l2*l2)/(2.*l1*l2)]
                            ang23+=[(l1*l1-l3*l3-l2*l2)/(2.*l3*l2)]
                            angmu+=[theta[k]]
            #array of length 3*number of triangles  
        
            ang12=np.array(ang12)
            ang23=np.array(ang23)
            ang31=np.array(ang31)
            angmu=np.array(angmu)  
          
            pickle.dump([ang12,ang23,ang31,angmu],open(filename_ang, 'w'))
            pickle.dump(ell,open(filename, 'w'))
        ell=np.asarray(ell)
        print "ell_type: %s"%ell_type
        
        if not (cross_bias or equilat or folded):
            ff_name     = path+"Ll_file_%s_%.0e_%d_lenL%d_lenang%d_%.0e.pkl"%(ell_type,l_min,l_max,len_L,len_ang,Delta_theta)
            pickle.dump(ell[1::3],open(ff_name,'w'))
    
    
        if kkg:
            config = 'kkg_%s'%ell_type
        elif kgg:
            config = 'kgg_%s'%ell_type
        else:
            config = ell_type
        if LSST:
            config+='bin_%s_%s'%(red_bin,dn_filename)
        else:
            config+='no_binning'
        
        config+=spectrum_config
       
        if B_fit:
            config+="_Bfit"
        
        config +="_"+params[0]['name']
        
        config+=tag
            
        print "config: %s"%config
            
        pickle.dump([cosmo.class_params],open('class_settings_%s.pkl'%config,'w'))
     
        bs   = Bispectra(cosmo,data,ell,z,config,ang12,ang23,ang31,path,z_cmb, bias, nl,B_fit,kkg, kgg, dndz, norm,k_min,k_max,sym,fit_z_max,cross_bias)
        bs()  
        
        if integrals:
            Int0 = I0(bs.bi_phi, bs.ell, angmu, len_L, len_l, len_ang, fullsky=False)
                
            Int2 = I2(bs.bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)
            
            Bi_cum=[]
            
            for R in rad:
                Bi_cum+=[skew(bs.bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, R=R,fullsky=False)]
         
            L    = np.unique(ell[0::3])
            
            if sym:
                config+='_sym'
        
            pickle.dump([params,Limber,L,Int0,Int2,rad,np.asarray(Bi_cum)],open('./cross_integrals/I0I1I2%s.pkl'%(config),'w'))
        
        if post_born:
            print 'computing post Born corrections...'
            config +='_postBorn'
            if bs.set_stage==False:
                bs.set_up()
            k_min   = bs.kmin
            k_max   = bs.kmax
            PBB     = postborn.PostBorn_Bispec(cosmo.class_params,k_min,k_max,kkg, dndz,norm)
            ell     = np.asarray(ell)
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
            print 'Done!'
    
            if integrals:
                Int0 = I0(bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)
                    
                Int2 = I2(bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)
             
                L    = np.unique(ell[0::3])
            
                pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s.pkl'%(config),'w'))                
                Int0 = I0(bi_kkg, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)
                    
                Int2 = I2(bi_kkg, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)
            
                pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s_only.pkl'%(config),'w'))
            
        del bs
        try:
            del bi_phi
        except:
            pass
    
        

            



           

      
