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

from N32biasIntegrals import I0, I1, I2
#from BispectrumIntegrals import integrate_bispec
import os
import pickle



class Bispectra():
    """ Bispectra of 
        - delta (function of chi)
        - newtonian potential bi_psi (function of chi)
        - lensing potential bi_phi
    """
    def __init__(self,cosmo,data, ell,z,config,ang12,ang23,ang31, path, z_cmb, nonlin=False, B_fit=False, dndz=None,norm=None,k_min=None,k_max=None):
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
  
        self.ell_min    = min(ell[::3])
        self.ell_max    = max(ell[::3])
        self.bin_num    = len(ell)
        self.len_bi     = self.bin_num/3
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
        self.ang31      = ang31
        
        if dndz ==None:
            self.cross=False
        else:
            self.cross=True
            self.dndz = dndz
            self.norm=norm
        
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
            
        self.filename=self.path+"bispec_phi_%s_lmin%d-lmax%d-lenBi%d"%(self.config,self.ell_min,self.ell_max,self.len_bi)
          
        try:
            self.bi_phi=np.load(self.filename+'.npy')
            print "loading file %s"%(self.filename+'.npy')									
        except:
            print "%s not found \n Computing Bispectrum of overdensity..."%self.filename
            self.filename2=self.path+"bispec_delta_%s_lmin%d-lmax%d-lenBi%d"%(self.config,self.ell_min,self.ell_max,self.len_bi)
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
        if self.cosmo.class_params['output']!='tCl, lCl, mPk':
            self.cosmo.class_params['output']='tCl, lCl, mPk'
            self.cosmo.class_params['lensing']='yes'
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
            cl_unl=self.closmo_lin.raw_cl(self.ell_max)
            cl_len=self.closmo_lin.lensed_cl(self.ell_max)
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
            cl_unl=self.closmo_nl.raw_cl(self.ell_max)
            cl_len=self.closmo_nl.lensed_cl(self.ell_max)
        else:
            self.closmo_nl=None
            
        self.set_stage=True
        try:
            print self.cosmo.class_params['A_s']
        except:
            print self.cosmo.class_params['ln10^{10}A_s']
            
        pickle.dump([self.cosmo.class_params,cl_unl,cl_len],open('../class_outputs/class_cls_%s.pkl'%self.cosmo.tag,'w'))
        
            
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
																				
            bi_delta[i] = bi_delta_chi#*self.delta2psi(k_i,len(bi_delta_chi),i)
                   
        self.bi_delta=np.transpose(bi_delta) #row is now a function of chi
        del self.k
        
        

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
        B+=2.*hf.get_F2_kernel(k1,k3,self.ang31)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
#        index   =np.where(np.any([(k1>self.kmax),(k1<self.kmin)],axis=0))
#        B[index]=0.
#        index   =np.where(np.any([(k2>self.kmax),(k2<self.kmin)],axis=0))
#        B[index]=0.
#        index   =np.where(np.any([(k3>self.kmax),(k3<self.kmin)],axis=0))
#        B[index]=0.
        						
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
        B+=2.*self.get_F2_kernel_fit(k1,k3,self.ang31,i)*np.exp(spec(np.log(k3)))*np.exp(spec(np.log(k1)))
        
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
            if self.cross:
                alpha       = np.ones(shape)*self.data.Poisson_prefac[i]**2
                alpha      *= 1./(k[::3]*k[2::3])**2               
            else:
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
            self.set_up()
        
        index   = np.arange(self.len_bi)
        

        W_lens  = ((self.chi_cmb-self.chi)/(self.chi_cmb*self.chi))*(self.z+1.)
        dchidz  = self.data.dchidz(self.z)
        dzdchi  = 1./dchidz
        if self.cross:
            W_gal   = (self.z+1.)*self.dndz(self.z)/self.norm*dzdchi #b=(1+z)
            kernel  = W_gal*W_lens**2
        else:
            kernel  = W_lens**3*self.chi**2

        bi_phi=[]
        for j in index:
            integrand   = self.bi_delta[j]*kernel
            bi_phi+=[simps(integrand,self.chi)]
        self.bi_phi=np.array(bi_phi)
        
        if self.cross:
            fac=self.data.prefacs**2*((1./((self.ell[0::3]+0.5)*(self.ell[1::3]+0.5)))**2+(1./((self.ell[1::3]+0.5)*(self.ell[2::3]+0.5)))**2+(1./((self.ell[2::3]+0.5)*(self.ell[0::3]+0.5)))**2)
            self.bi_phi*=fac
        else:
            fac=(self.data.prefacs)**3/((self.ell[1::3]+0.5)*(self.ell[2::3]+0.5)*(self.ell[0::3]+0.5))**2
            self.bi_phi*=fac
        return True
        



"""---- Choose you settings here ---"""
if __name__ == "__main__":  
    
    "---begin settings---"
    cross       = True 

    if cross:   
        dn_filename = 'dndz_LSST_i27_SN5_3y'
    
    #choose Cosmology (see Cosmology module)
    params      = C.Planck2015_TTlowPlensing

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = False    
    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662
    post_born   = True
    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = True
    # compute C^(phi,g)
    cross_spec  = False
    pow_spec    = False
    #binbounds
    
    red_bin     = '0'
    
    bounds      = {'0':[0.0,0.5],'1':[0.5,1.],'2':[1.-2.]}
				    
    #number of redshift bins 
    bin_num     = 300
    
    #sampling in L/l and angle
    len_L       = 163
    len_ang     = 163

    #ell range (for L and l)
    ell_min     = 2.
    ell_max     = 3000.
    
    #tag for L-sampling
    ell_type    ="linlog_newang"
    
    #regularizing theta bounds
    Delta_theta = 1e-2
    
    nl          = True
    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'
    #path, where to store results
    path            = "/afs/mpa/temp/vboehm/spectra/"
    
    "---end settings---"
    if cross: 
        gz, dgn = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
        
    #initialize cosmology
    cosmo   = C.Cosmology(zmin=0.00, zmax=1200, Params=params, Limber=Limber, lmax=ell_max, mPk=False, Neutrinos=False)
    closmo  = Class()
    closmo.set(params[1])
    closmo.compute()
    #set up z range and binning in z space
    z_min   = 1e-3
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    
    print "z_cmb: %f"%z_cmb

    #linear sampling in z is ok
    z       = np.exp(np.linspace(np.log(z_min),np.log(z_cmb-0.01),bin_num))
    print max(z)
    if cross:
        z       = np.linspace(max(bounds[red_bin][0],z_min),bounds[red_bin][1],100)
        dndz    = interp1d(gz, dgn, kind='linear')
        norm    = simps(dndz(z),z)
    else:
        dndz = None
        norm = None
    #cosmo dependent functions
    data    = C.CosmoData(cosmo,z)
    print z
    #list of all triangles and their sides (loading is faster than recomputing)
    filename=path+"ell_%s_%.0e_%d_lenL%d_lenang%d_%.0e.pkl"%(ell_type,ell_min,ell_max,len_L,len_ang,Delta_theta)
    filename_ang=path+"ang_%s_%.0e_%d_lenL%d_lenang%d_%.0e.pkl"%(ell_type,ell_min,ell_max,len_L,len_ang,Delta_theta)
    
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
        if ell_type=="linlog_newang":
            #L = |-L|, equally spaced in lin at low L and in log at high L 
            La        = np.linspace(ell_min,50,48,endpoint=False)
            Lb        = np.exp(np.linspace(np.log(50),np.log(ell_max),len_L-48))
            L         = np.append(La,Lb)
            #l 
            l         = L  
        elif ell_type=="lin":
            #L = |-L|, equally spaced in lin at low L and in log at high L 
            L         = np.linspace(ell_min,ell_max)
            l         = L
        else:
            raise Exception("ell type not consistent with any sampling method")
            
        # angle, cut edges to avoid numerical instabilities
        #TODO: try halving this angle, probably requires multiplication by 2, but should avoid l2=0
        theta   = np.linspace(Delta_theta,2*np.pi-Delta_theta, len_ang)
        cosmu   = np.cos(theta) #Ldotl/Ll or -l1dotl3/l1/l3 (l1+l2+l3=0) (angle used in beta Integrals)
        
        ang31=[]
        ang12=[]
        ang23=[]
        angmu=[]

        sqrt=np.sqrt
        #all combinations of the two sides and the angles
        for i in range(len_L):
            for j in range(len_L):
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
    
    ff_name     = path+"Ll_file_%s_%.0e_%d_lenL%d_lenang%d_%.0e.pkl"%(ell_type,ell_min,ell_max,len_L,len_ang,Delta_theta)
    pickle.dump(ell[1::3],open(ff_name,'w'))

    if cross:
        config = 'cross_g_bin%s'%red_bin+ell_type
    else:
        config = ell_type
    
    config+=spectrum_config
   
    if B_fit:
        config+="_Bfit"
    
    config +="_"+params[0]['name']
        
    print "config: %s"%config
        
    pickle.dump([cosmo.class_params],open('class_settings_%s.pkl'%config,'w'))
 
    bs   = Bispectra(cosmo,data,ell,z,config,ang12,ang23,ang31,path,z_cmb, nl,B_fit,dndz,norm)
    bs()  
    
    Int0 = I0(bs.bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
    
    Int1 = I1(bs.bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
        
    Int2 = I2(bs.bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
 
    L    = np.unique(ell[0::3])

    pickle.dump([params,Limber,L,Int0,Int1,Int2],open('I0I1I2%s.pkl'%(config),'w'))
    
    if post_born:
        print 'computing post Born corrections...'
        config +='_postBorn'
        if bs.set_stage==False:
            bs.set_up()
        k_min   = bs.kmin
        k_max   = bs.kmax
        PBB     = postborn.PostBorn_Bispec(cosmo.class_params,k_min,k_max,cross, dndz)
        ell     = np.asarray(ell)
        if cross:
            try:
                bi_cross_sum = np.load(bs.filename+"_post_born_sum.npy")
                bi_cross     = np.load(bs.filename+"_post_born.npy")
            except:
                bi_cross = PBB.bi_born_cross(ell[0::3],ell[1::3],ell[2::3],16./(3*data.Omega_m0*data.H_0**2))
                bi_cross_sum = bi_cross+bs.bi_phi
                np.save(bs.filename+"_post_born.npy",bi_cross)
                np.save(bs.filename+"_post_born_sum.npy",bi_cross_sum)
            bi_phi = bi_cross_sum
        else:
            bi_post  = (PBB.bi_born(ell[0::3],ell[1::3],ell[2::3])*8./(ell[0::3]*ell[1::3]*ell[2::3])**2)
            np.save(bs.filename+"_post_born.npy",bi_post)
            np.save(bs.filename+"_post_born_sum.npy",bi_post+bs.bi_phi)
            bi_phi = bi_post+bs.bi_phi
        print 'Done!'

        Int0 = I0(bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
        
        Int1 = I1(bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
            
        Int2 = I2(bi_phi, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
     
        L=np.unique(ell[0::3])

        pickle.dump([params,Limber,L,Int0,Int1,Int2],open('I0I1I2%s.pkl'%(config),'w'))
        
        Int0 = I0(bi_cross, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
        
        Int1 = I1(bi_cross, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
            
        Int2 = I2(bi_cross, bs.ell, angmu ,len_L*len_ang, len_L, squeezed=False, fullsky=False)
     
        L=np.unique(ell[0::3])

        pickle.dump([params,Limber,L,Int0,Int1,Int2],open('I0I1I2%s.pkl'%(config+'only'),'w'))


#    Ints    = integrate_bispec(bs.bi_phi, ell, angmu, len_L, len_ang, fullsky=False)
#
#    L       = np.unique(ell[0::3])
#    
#    pickle.dump([L,Ints],open('bispectrum_integrals_%s.pkl'%config,'w'))
#    
##    L_, Int0_old, dum1, dum2= pickle.load(open('I0I1I2linlog_full_linPS_Planck2013.pkl','r'))
###
#    labels=['Ia', 'Ib', 'IIa', 'IIb', 'IIc']
#    colors=['b', 'g', 'r', 'c', 'm']
#    pl.figure()
#    for m in range(5):
#        try:
#            pl.semilogy(L,L**2*Ints[m],ls='-',color=colors[m],label=labels[m])
#        except:
#            pl.semilogy(L,-L**2*Ints[m],ls='--',color=colors[m],label=labels[m])
#    pl.legend()
#    pl.xlim([50,3000])
#    pl.xticks([50,500,1000,2000,3000])
#    pl.ylabel(r'$L^2$ Integrals')
#    pl.xlabel(r'$L$')
#    pl.grid()
#    pl.savefig('bispectrum_integrals.png')
#    pl.close()
#    
#    pl.figure()
#    for m in [1,4]:
#        print Ints[m]
#        try:
#            pl.semilogy(L,-L**2*Ints[m],ls='--',color=colors[m],label=labels[m])
#        except:
#            pl.semilogy(L,L**2*Ints[m],ls='-',color=colors[m],label=labels[m])
#    pl.legend()
#    pl.xlim([50,3000])
#    pl.xticks([50,500,1000,2000,3000])
#    pl.ylabel(r'$L^2$ Integrals')
#    pl.xlabel(r'$L$')
#    pl.savefig('vanishing_bispectrum_integrals.png')
#    
    
        

            



           

      
