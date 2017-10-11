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
import CAMB_postborn as postborn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.ioff()

from classy import Class
import Cosmology as C

from BispectrumIntegrals import integrate_bispec
import os
import pickle



class Bispectra():
    """ Bispectra of 
        - delta (function of chi)
        - newtonian potential bi_psi (function of chi)
        - lensing potential bi_phi
    """
    def __init__(self,cosmo,data, ell,z,config,ang12,ang23,ang13, path, nonlin=False, B_fit=False, k_min=None,k_max=None):
        """
        initializes/computes all three bispectra
        * cosmo:    instance of class Cosmology
        * data:     instance of class CosmoData
        * ell:      array of ell vector absolute values
        * z:        array of redshifts
        * config:   string encoding the configuration of this run
        * ang12,23,13: array of angles between vectors
        """
        print "nl", nonlin
        print "\n"

        self.cosmo      = cosmo
        
        self.ell        = ell    
        
        self.data       = data
        #for comoving angular diameter distance
        self.chi        = self.data.chi(z)
        self.chi_cmb    = max(self.chi)
        print "chi_cmb: %f"%self.chi_cmb
        self.z          = z
        assert((data.z==self.z).all())
  
        self.ell_min    = min(ell)
        self.ell_max    = max(ell)
        self.bin_num    = len(ell)
        self.len_bi     = self.bin_num/3.
        print "ell min: ", self.ell_min
        print "ell max: ", self.ell_max
        print "bispectrum size: ", self.len_bi
        
        self.kmin       = k_min
        self.kmax       = k_max
        
        self.config     = config
        print "configuration: ", self.config
        #has set_up been called?
        self.set_stage  = False
        
        self.ang12      = ang12
        self.ang23      = ang23
        self.ang13      = ang13
        
        self.nl         = nonlin
        if self.nl:
            print "Using non-linear matter power spectrum"	


        self.B_fit  = B_fit
        if self.B_fit:
            print "using Gil-Marin et al. fitting formula"
        
        self.path   = path
        
        
        
    def __call__(self):
        """
        call method of bispectrum class
        computes the lensing bispectrum
        """
            
        self.filename=self.path+"bispec_phi_%s_lmin%d-lmax%d-lenBi%d_1e-2"%(self.config,self.ell_min,self.ell_max,self.len_bi)
          
        try:
            self.bi_phi=np.load(self.filename+".npy")
            print "loading file %s"%self.filename												
        except:
            print "%s not found \n Computing Bispectrum of Lensing Potential..."%self.filename
            filename2=self.path+"bispec_psi_%s_lmin%d-lmax%d-lenBi%d_1e-2"%(self.config,self.ell_min,self.ell_max,self.len_bi)
            try:
                self.bi_psi=np.load(filename2)
            except:
                print "%s not found \n Computing Bispectrum of Newtonian Potential..."%self.filename
                
                self.compute_Bispectrum_Psi()
                #np.save(filename2+".npy",self.bi_psi)

            self.compute_Bispectrum_Phi()
            np.save(self.filename+".npy",self.bi_phi)
                
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
        self.pow_ell     = np.linspace(min(self.ell)/10.,max(self.ell)+1,1000)
        #number of unique l values
        self.powspec_len = np.int(len(self.pow_ell))

        #k=ell/chi, ndarray with ell along rows, chi along columns
        ell     = np.sqrt(self.ell*(self.ell+np.ones(len(self.ell))))
        self.k  = np.outer(1./self.chi,ell)
								
        kmax    = max(self.k.flatten())
        kmin    = min(self.k.flatten())
        				
        print "kmin and kmax from ell/chi", kmin,kmax

        if self.B_fit:
            k4n=np.exp(np.linspace(np.log(kmin),np.log(kmax),200))
            k4n=np.concatenate((k4n,np.exp(np.linspace(-4,-1,200))))
            k4n=np.sort(k4n)
            self.data.get_abc(k4n,self.z)
        if self.kmin==None:
            self.kmin=kmin
        if self.kmax==None:
            self.kmax=kmax
        print "kmin and kmax for bispectrum calculation", self.kmin,self.kmax 
        if self.cosmo.class_params['output']!='tCl, mPk':
            self.cosmo.class_params['output']='tCl, mPk'
            self.cosmo.class_params['tol_perturb_integration']=1.e-6
								
        self.cosmo.class_params['z_max_pk'] = max(z)		
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
            cl_unl=self.closmo_lin.raw_cl(4000)
            print self.closmo_lin.get_current_derived_parameters(['sigma8'])
        else:
            self.closmo_lin=None

        if self.nl:
            self.cosmo.class_params['non linear'] = "halofit"
            self.closmo_nl=Class()
            self.closmo_nl.set(self.cosmo.class_params)
            print "Initializing CLASS with halofit..."
            print self.cosmo.class_params
            self.closmo_nl.compute()
            print self.closmo_nl.get_current_derived_parameters(['sigma8'])
            cl_unl=self.closmo_nl.raw_cl(4000)
        else:
            self.closmo_nl=None
        
        self.set_stage=True
        try:
            print self.cosmo.class_params['A_s']
        except:
            print self.cosmo.class_params['ln10^{10}A_s']
            
        pickle.dump([self.cosmo.class_params,cl_unl],open('../class_outputs/class_cls_%s.pkl'%self.cosmo.tag,'w'))
        
            
    def compute_Bispectrum_Psi(self):
        """ 
        computes the bispectrum for each chi bin
        -> only use after self.set_up() has been called! 
        """      
        if self.set_stage==False:
            self.set_up()
        
        bi_psi  = np.ndarray((len(self.z),self.len_bi))
     
        #more exact Limber, k for power spectrum
        ell     = np.sqrt(self.pow_ell*(self.pow_ell+np.ones(len(self.pow_ell))))
        
        if self.nl:
            cosmo_pk = self.closmo_nl.pk
        else:
            cosmo_pk = self.closmo_lin.pk
        
        for i in np.arange(0,len(self.z)):
            
            print i
            
            z_i    = self.z[i]
            k_i    = self.k[i]
            
            k_spec = (ell/self.chi[i])
            spec=[]

            for j in np.arange(0,len(k_spec)):
                spec+=[cosmo_pk(k_spec[j],z_i)]
            spec=np.array(spec)
            
            if self.B_fit==False:
                    bi_delta_chi    = self.bispectrum_delta(spec,k_spec,k_i)
                    
            elif self.B_fit and self.z[i]<=1.5:
                    bi_delta_chi    = self.bispectrum_delta_fit(spec,k_spec,k_i,i)
            elif self.B_fit and self.z[i]>1.5:
                    bi_delta_chi    = self.bispectrum_delta(spec,k_spec,k_i)
            else:
                raise ValueError('Something went wrong with matter bispectrum specifications')
																				
            bi_psi[i] = bi_delta_chi*self.delta2psi(k_i,len(bi_delta_chi),i)
                   
        self.bi_psi=np.transpose(bi_psi) #row is now a function of chi
        
        

    def bispectrum_delta(self,spectrum,k_spec,k):
        """ returns the bispectrum of the fractional overdensity today (a=1) i.e. B^0, the lowest order in non-lin PT
        *spectrum:   power spectrum for all ks in k_aux
        *k_spec:     array of ks where for which power spectrum is passed
        *k:          array of k's that form the triangles for which the bispectrum is computed
        """
        spec     = interp1d(np.log(k_spec),np.log(spectrum),kind="slinear")
    
        k1       = k[::3]
        k2       = k[1::3]
        k3       = k[2::3]

        B=2.*hf.get_F2_kernel(k1,k2,self.ang12)*np.exp(spec(np.log(k1)))*np.exp(spec(np.log(k2)))
        B+=2.*hf.get_F2_kernel(k2,k3,self.ang23)*np.exp(spec(np.log(k2)))*np.exp(spec(np.log(k3)))
        B+=2.*hf.get_F2_kernel(k1,k3,self.ang13)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
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
        spec  = interp1d(np.log(k_spec),np.log(spectrum),kind="slinear")
		
        k1       = k[::3]
        k2       = k[1::3]
        k3       = k[2::3]

        B= 2.*self.get_F2_kernel_fit(k1,k2,self.ang12,i)*np.exp(spec(np.log(k1)))*np.exp(spec(np.log(k2)))
        B+=2.*self.get_F2_kernel_fit(k2,k3,self.ang23,i)*np.exp(spec(np.log(k2)))*np.exp(spec(np.log(k3)))
        B+=2.*self.get_F2_kernel_fit(k1,k3,self.ang13,i)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
#        index=np.where(np.any([(k1>self.kmax),(k1<self.kmin)],axis=0))
#        B[index]=0.
#        index=np.where(np.any([(k2>self.kmax),(k2<self.kmin)],axis=0))
#        B[index]=0.
#        index=np.where(np.any([(k3>self.kmax),(k3<self.kmin)],axis=0))
#        B[index]=0.

        return B
        
 
    def get_F2_kernel_fit(self,k1,k2,cos,i):

        ak1=splev(k1, self.data.a_nk[i])
        ak2=splev(k2, self.data.a_nk[i])
        
        bk1=splev(k1, self.data.b_nk[i])
        bk2=splev(k2, self.data.b_nk[i])
        
        ck1=splev(k1, self.data.c_nk[i])
        ck2=splev(k2, self.data.c_nk[i])
        
        a=5./7.*ak1*ak2 #growth
        b=0.5*(k1/k2+k2/k1)*cos*bk1*bk2 #shift
        c=2./7.*cos**2*ck1*ck2 #tidal
    
        F2=a+b+c
        
        return F2
        
        
    def delta2psi(self,k,shape,i,bi_spec=True):
        """ returns the factor that converts the bispectrum in delta to the bispectrum in psi at given chi
        * k:        array of k at fixed chi
        * shape:    shape of delta bispectrum
        * i:        index in chi  
        * bi_spec:  whether to return the delta2psi prefactor for the bispectrum (=True) or the power spectrum (=False)
        * bi_spec:  if False, equivalent operation for power spectrum is performed
        """
        if bi_spec:
            alpha       = np.ones(shape)*self.data.Poisson_prefac[i]**3
            alpha      *= 1./(k[::3]*k[1::3]*k[2::3])**2
        else:
            alpha       = np.ones(shape)*self.data.Poisson_prefac[i]**2
            alpha      *= 1./(k)**4
        
        return alpha
   
    def compute_Bispectrum_Phi(self):
        """ computes the bispectrum of the lensing potential 
        Computes the bispectrum by integration over chi for ever triangle
        """
        if self.set_stage==False:
            self.set_up(self.ell)
        
        index   = np.arange(self.len_bi)
        
        W       = -2.*(self.chi_cmb-self.chi)/(self.chi_cmb*self.chi)
        
        kernel  = W**3/(self.chi**4)

        bi_phi=[]
        for j in index:
            integrand   = self.bi_psi[j]*kernel
            bi_phi+=[simps(integrand,self.chi)]
        self.bi_phi=np.array(bi_phi)  

        return True

        


"""---- Choose you settings here ---"""
if __name__ == "__main__":  
    
    "---begin settings---"
    #choose Cosmology (see Cosmology module)
    params      = C.Planck2013_TempLensCombined

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = True
    
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662
    post_born   = True
    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
				    
    #number of redshift bins 
    bin_num     = 594
    
    #sampling in L/l and angle
    len_L       = 163
    len_ang     = 163

    #ell range (for L and l)
    ell_min     = 0.1
    ell_max     = 8000.
    
    #tag for L-sampling
    ell_type    ="linlog"
				
    #for simulation only
    #k_min       = 0.0105*params[1]['h'] #h/Mpc
    #k_max       = 49*params[1]['h']
        
    #special configurations, eg which power spectrum is used in B_delta
    nl              = False      
    spectrum_config = '_linPS'
    
    #path, where to store results
    path            = "/afs/mpa/temp/vboehm/spectra/"
    
    "---end settings---"
    

        
    #initialize cosmology
    cosmo   = C.Cosmology(zmin=0.00, zmax=1200, Params=params, flatsky=Limber, lmax=ell_max, mPk=False, Neutrinos=False)
    closmo  = Class()
    closmo.set(params[1])
    closmo.compute()
    #set up z range and binning in z space
    z_min   = 0.01
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    
    print "z_cmb: %f"%z_cmb

    #linear sampling in z is ok
    z       = np.linspace(z_min,np.round(z_cmb,2),bin_num)
    print max(z)

    #cosmo dependent functions
    data    = C.CosmoData(cosmo,z)

    #list of all triangles and their sides (loading is faster than recomputing)
    filename=path+"ell_%s_%d_%d_lenL%d_lenang%d_1e-2.pkl"%(ell_type,ell_min,ell_max,len_L,len_ang)
    filename_ang=path+"ang_%s_%d_%d_lenL%d_lenang%d_1e-2.pkl"%(ell_type,ell_min,ell_max,len_L,len_ang)
        
    print filename
        
    try:
        ell=pickle.load(open(filename))
        ell=ell[0]
        angles=pickle.load(open(filename_ang))
        ang12=angles[0]
        ang23=angles[1]
        ang13=angles[2]
        angmu=angles[3]
    except:
        ell         = []
        print "ell file not found"
        if ell_type=="linlog":
            #L = |-L|, equally spaced in lin at low L and in log at high L 
            La        = np.linspace(ell_min,50,48,endpoint=False)
            Lb        = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
            L         = np.append(La,Lb)
            #l 
            la        = np.linspace(ell_min,50,48,endpoint=False)
            lb        = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
            l         = np.append(la,lb)
        elif ell_type=="lin":
            #L = |-L|, equally spaced in lin at low L and in log at high L 
            L         = np.linspace(ell_min,ell_max)
            l         = L
        else:
            raise Exception("ell type not consistent with any sampling method")
            
        #print "L", L
        #print "l", l
        # angle, cut edges to avoid numerical instabilities
        theta   = np.linspace(1e-2,2*np.pi-1e-2, len_ang)
        cosgam  = np.cos(theta) #Ldotl/Ll
        
        ang13=[] #-Ldotl/Ll
        ang12=[] #-Ldot(L-l)/l/|L-l|
        ang23=[] #ldot(L-l)/l/|L-l|
        angmu=[] #Ldotl/Ll
        #all combinations of the two sides and the angles
        for i in range(len_L):
            for j in range(len_L):
                for k in range(len_ang):
                    Ll=hf.coslaw_side(L[i],l[j],cosgam[k]) #|L-l|=L**2+l**2-2Ldotl
                    if Ll<1e-5:
                        Ll = 1e-5
                    ell+=[L[i]]+[Ll]+[l[j]]
                    mLdotl=-L[i]*l[j]*cosgam[k]
                    Ldotl = L[i]*l[j]*cosgam[k]
                    Ldotl2=-L[i]**2.+Ldotl #-L*(L-l)
                    ldotl2=-l[j]**2.+Ldotl # l(L-l)
                    ang13+=[(mLdotl/L[i]/l[j])]
                    ang12+=[(Ldotl2/L[i]/Ll)]
                    ang23+=[(ldotl2/l[j]/Ll)]
                    angmu+=[theta[k]]
        #array of length 3*number of triangles  
    
        ang12=np.array(ang12)
        ang23=np.array(ang23)
        ang13=np.array(ang13)
        angmu=np.array(angmu)
        
      
        pickle.dump([ang12,ang23,ang13,angmu],open(filename_ang, 'w'))
        pickle.dump([ell],open(filename, 'w'))
    

    print "ell_type: %s"%ell_type

    config = ell_type+spectrum_config
   
    if B_fit:
        config+="_Bfit"

    
    config = config+"_"+params[0]['name']
        
    print "config: %s"%config
        
    pickle.dump([cosmo.class_params],open('class_settings_%s.pkl'%config,'w'))

    bs      = Bispectra(cosmo,data,ell,z,config,ang12,ang23,ang13,path,nl,B_fit)
    bs()
    
    if post_born:
        #bs.set_up()
        k_min=bs.kmin
        k_max=bs.kmax
        PBB=postborn.PostBorn_Bispec(cosmo.class_params,k_min,k_max)
        ell=np.asarray(ell)
        bi_post=(PBB.bi_born(ell[0::3],ell[1::3],ell[2::3])*8./(ell[0::3]*ell[1::3]*ell[2::3])**2)
        np.save(bs.filename+"_cut_post_born.npy",bi_post)
        np.save(bs.filename+"_cut_post_born_sum.npy",bi_post+bs.bi_phi)

    Ints    = integrate_bispec(bs.bi_phi, ell, angmu, len_L, len_ang, fullsky=False)

    L       = np.unique(ell[0::3])
    
    pickle.dump([L,Ints],open('bispectrum_integrals_%s_3.pkl'%config,'w'))
    
#    L_, Int0_old, dum1, dum2= pickle.load(open('I0I1I2linlog_full_linPS_Planck2013.pkl','r'))
##
    labels=['Ia', 'Ib', 'IIa', 'IIb', 'IIc']
    colors=['b', 'g', 'r', 'c', 'm']
    pl.figure()
    for m in range(5):
        try:
            pl.semilogy(L,L**2*Ints[m],ls='-',color=colors[m],label=labels[m])
        except:
            pl.semilogy(L,-L**2*Ints[m],ls='--',color=colors[m],label=labels[m])
    pl.legend()
    pl.xlim([50,3000])
    pl.xticks([50,500,1000,2000,3000])
    pl.ylabel(r'$L^2$ Integrals')
    pl.xlabel(r'$L$')
    pl.grid()
    pl.savefig('bispectrum_integrals.png')
    pl.close()
    
    pl.figure()
    for m in [1,4]:
        print Ints[m]
        try:
            pl.semilogy(L,-L**2*Ints[m],ls='--',color=colors[m],label=labels[m])
        except:
            pl.semilogy(L,L**2*Ints[m],ls='-',color=colors[m],label=labels[m])
    pl.legend()
    pl.xlim([50,3000])
    pl.xticks([50,500,1000,2000,3000])
    pl.ylabel(r'$L^2$ Integrals')
    pl.xlabel(r'$L$')
    pl.savefig('vanishing_bispectrum_integrals.png')
    
    
        

            



           

      
