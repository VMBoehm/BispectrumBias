# -*- coding: utf-8 -*-
"""
Created on 12.02.2015

@author: Vanessa Boehm

Cosmology.py:
	* Cosmological Parameter sets
	* Class Cosmology()
	* Class CosmoData()
"""

from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.ioff()
import copy

import Constants as const
import HelperFunctions as HF
from classy import Class
import pickle


#TODO: Before using again, check if correct neutrino parameters are set
#""" Planck 2013 Cosmological parameters from CMB and CMB temperature lensing + neutrinos """        
#Planck2013_TempLensCombinedNeutrinos={
#        'h': 0.6712,
#        'omega_b' : 0.022242,
#        'omega_cdm': 0.11805,
#        'Omega_k' : 0.0,
#        'tau_reio' : 0.0949,
#        'A_s'    : 2.215*1e-9,
#        'n_s'    : 0.9675,
#        'N_ur'   : 2.0328, #two massless, one massive neutrino, as base Lambda CDM
#        'N_ncdm' : 1,
#        'm_ncdm' : '0.06'}

#""" Fiducial Cosmology used in the Planck Lensing Analysis: http://arxiv.org/abs/1502.01591 """
#PlanckLensingFiducial2015={
#'name':"PlanckLensingFiducial2015",
#'h': 0.6712,
#'omega_b' : 0.0222,
#'omega_cdm': 0.1203,
#'Omega_k' : 0.0,
#'tau_reio' : 0.065,
#'A_s'    : 2.09*1e-9,
#'n_s'    : 0.96,
#'N_ur' : 2.0328, #two massless, one massive neutrino
#'N_ncdm' : 1,
#'m_ncdm' : '0.06'}

""" Planck 2013 Cosmological parameters from CMB and CMB temperature lensing, no neutrinos """               
Planck2013_TempLensCombined=[{
'name':"Planck2013_TempLensCombined"},{
'h': 0.6714,
'omega_b' : 0.022242,
'omega_cdm': 0.11805,
'Omega_k' : 0.0,
'tau_reio' : 0.0949,
'A_s'    : 2.215*1e-9,
'n_s'    : 0.9675,
'k_pivot' : 0.05}]

ToshiyaComparison=[{
'name':"Toshiya"},{
'h': 0.6751,
'omega_b':0.0223,
'omega_cdm':0.119,
'Omega_k':0.0,
'tau_reio':0.063,
'A_s'   :2.13e-9,
'n_s'   :0.965,
#'N_ncdm': 2,
#'N_ur':1.0196,
#'m_ncdm': "0.05, 0.01",
'k_pivot' : 0.05,
'tau_reio':0.0630,
#'ncdm_fluid_approximation': 2,
#'ncdm_fluid_trigger_tau_over_tau_k':51.,
#'tol_ncdm_synchronous':1.e-10,
#'tol_ncdm_bg':1.e-10,
#'l_max_ncdm':51
}]
               
Namikawa=[{
'name':"Namikawa_Paper"},{
'h': 0.6712,
'omega_b':0.0223,
'omega_cdm':0.119,
'Omega_k':0.0,
'tau_reio':0.0630,
'A_s'   :2.13e-9,
'n_s'   :0.965,
'N_ncdm': 2,
'N_ur':1.0196,
#'m_ncdm': "0.05, 0.01",
'k_pivot' : 0.05,
'tau_reio':0.0630,
#'ncdm_fluid_approximation': 2,
#'ncdm_fluid_trigger_tau_over_tau_k':51.,
#'tol_ncdm_synchronous':1.e-10,
#'tol_ncdm_bg':1.e-10,
#'l_max_ncdm':51
}]

Pratten=[{
'name':"Pratten_Paper"},{
'h': 0.67,
'omega_b':0.022,
'omega_cdm':0.122,
'Omega_k':0.0,
'tau_reio':0.06,
'A_s'   :2.0*1e-9,
'n_s'   :0.965,
'N_ncdm': 1,
'N_ur':2.0328,
'Omega_ncdm': 0.00064/(0.67**2),#m=0.06eV
'ncdm_fluid_approximation': 2,
'ncdm_fluid_trigger_tau_over_tau_k':51.,
'tol_ncdm_synchronous':1.e-10,
'tol_ncdm_bg':1.e-10,
'l_max_ncdm':51}]
#'N_ncdm':3,
#'N_ur':0.00641,
#'m_ncdm':"0.01, 0.05, 0.0",
#'tol_ncdm_bg':1.e-10,
#'recfast_Nz0':100000,
#'tol_thermo_integration':1.e-5,
#'recfast_x_He0_trigger_delta':0.01,
#'recfast_x_H0_trigger_delta':0.01,
#'evolver':0,
#'k_min_tau0':0.002,
#'k_max_tau0_over_l_max':3.,
#'k_step_sub':0.015,
#'k_step_super':0.0001,
#'k_step_super_reduction':0.1,
#'start_small_k_at_tau_c_over_tau_h':0.0004,
#'start_large_k_at_tau_h_over_tau_k':0.05,
#'tight_coupling_trigger_tau_c_over_tau_h':0.005,
#'tight_coupling_trigger_tau_c_over_tau_k':0.008,
#'start_sources_at_tau_c_over_tau_h':0.006,
#'l_max_g':50,
#'l_max_pol_g':25,
#'l_max_ur':150,
#'l_max_ncdm':50,
#'tol_perturb_integration':1.e-6,
#'perturb_sampling_stepsize':0.01,
#'radiation_streaming_approximation':2,
#'radiation_streaming_trigger_tau_over_tau_k':240.,
#'radiation_streaming_trigger_tau_c_over_tau':100.,
#'ur_fluid_approximation':2,
#'ur_fluid_trigger_tau_over_tau_k':50.,
#'ncdm_fluid_approximation':3,
#'ncdm_fluid_trigger_tau_over_tau_k':51.,
##'tol_ncdm':1.e-10,
#'l_logstep':1.026,
#'l_linstep':25,
#'hyper_sampling_flat':12.,
#'hyper_sampling_curved_low_nu':10.,
#'hyper_sampling_curved_high_nu':10.,
#'hyper_nu_sampling_step':10.,
#'hyper_phi_min_abs':1.e-10,
#'hyper_x_tol':1.e-4,
#'hyper_flat_approximation_nu':1.e6,
#'q_linstep':0.20,
#'q_logstep_spline':20.,
#'q_logstep_trapzd':0.5,
#'q_numstep_transition':250,
#'transfer_neglect_delta_k_S_t0':100.,
#'transfer_neglect_delta_k_S_t1':100.,
#'transfer_neglect_delta_k_S_t2':100.,
#'transfer_neglect_delta_k_S_e':100.,
#'transfer_neglect_delta_k_V_t1':100.,
#'transfer_neglect_delta_k_V_t2':100.,
#'transfer_neglect_delta_k_V_e':100.,
#'transfer_neglect_delta_k_V_b':100.,
#'transfer_neglect_delta_k_T_t2':100.,
#'transfer_neglect_delta_k_T_e':100.,
#'transfer_neglect_delta_k_T_b':100.,
#'neglect_CMB_sources_below_visibility':1.e-30,
#'transfer_neglect_late_source':3000.
#}]

        
""" Planck 2015 Cosmological parameters from different combinations of constraints"""
Planck2015_TTlowPlensing=[{
'name':"Planck2015_TTlowPlensing"},{
'h': 0.6781,
'omega_b' : 0.02226,
'omega_cdm': 0.1186,
'Omega_k' : 0.0,
'tau_reio' : 0.066,
'ln10^{10}A_s':3.062,
'n_s'    : 0.9677,
'k_pivot' : 0.05}]
        
Planck2015_TTTEEElowPlensing=[{
'name':"Planck2015_TTTEEElowPlensing"},{
'h': 0.6751,
'omega_b' : 0.02226,
'omega_cdm': 0.1193,
'Omega_k' : 0.0,
'tau_reio' : 0.063,
'ln10^{10}A_s': 3.059,
'n_s'    : 0.9653}]

""" Jia's and Colin's simulation """							
SimulationCosmology=[{'name':"Jias_Simulation"},{
'T_cmb':2.725,
'omega_cdm':0.129600,
'omega_b':0.023846,
'h':0.720,
'n_s':0.960,
'A_s':1.971*1e-9,
'Omega_k':0.0,
'k_pivot' : 0.002}]

""" Gil-Marin et al Simulations """							
BispectrumSimulations=[{'name':"Gil-Marin_et_al"},{
'T_cmb':2.725,
'omega_cdm':0.27*0.7**2-0.023,
'omega_b':0.023,
'h':0.7,
'n_s':0.95,
'A_s':2.585*1e-9,
'Omega_k':0.0,
'k_pivot' : 0.002}]

MatterOnly=[{'name':"Matter_Only"},{
'T_cmb':2.725,
'omega_cdm':0.5084,
'omega_b':0.01,
'h':0.720,
'n_s':0.960,
'A_s':1.971*1e-9,
'Omega_k':0.0,
'k_pivot' : 0.002}]


class Cosmology():
	""" cosmological parameter """
	def __init__(self, zmin=None, zmax=None, Params=None, Limber=False, lmax=None, mPk=False, Neutrinos=False,lensing=False):   
		
		if zmin==None:
			self.z_min      = 0.005
		else:
			self.z_min      = zmin
		print "starting integrations at zmin=%.4f"%self.z_min
		
		if zmax==None:
			self.z_max      = 1100
		else:
			self.z_max      = zmax
		print " zmax=%.1f"%self.z_max

		# parameters for class
		if Params!=None:
			print "cosmological parameter:",Params[0]['name']
			
			self.tag 		 = Params[0]['name']
			
			self.class_params = Params[1]
			
			print self.class_params
		else:
			self.class_params={}
			print "cosmological parameter: CLASS default" 
        	if lensing:
			self.class_params['lensing']='yes'

		if mPk:
			self.class_params['output']= 'tCl lCl mPk'
			self.class_params['tol_perturb_integration']=1.e-6
		else:
			self.class_params['output']= 'lCl tCl'
        
		if lmax==None:
			self.class_params['l_max_scalars']= 10000
		else:
			self.class_params['l_max_scalars']= lmax
			print 'l_max_scalars: ',self.class_params['l_max_scalars']
        
		if Limber:
			self.class_params['l_switch_limber']=1
		else:
			self.class_params['l_switch_limber']=100
			self.class_params['perturb_sampling_stepsize']=0.01
		print "Limber approximation from l=",self.class_params['l_switch_limber']
        
        
		if Neutrinos:
			self.class_params['ncdm_fluid_approximation'] = 3
			self.class_params['ncdm_fluid_trigger_tau_over_tau_k'] = 51.
			self.class_params['gauge']='synchronous'
			self.class_params['tol_ncdm_synchronous']=1.e-10
			self.class_params['tol_ncdm_bg']=1.e-10
			self.class_params['l_max_ncdm']=51
			print "precision settings set for accurate calcuclation of neutrino effects"


class CosmoData():
	""" class of z-dependent quantities (distances etc.) for a given cosmology """
	
	def __init__(self, cosmo, z, test=False):
		"""computes H(z), comoving distance, scale factor, prefactor in poisson equation as function of z
			* cosmo: 	instance of class Cosmology (paramter s for CLASS)
			* z: 	     	array of redshifts
		"""
		print "computing distances, derived parameters..." 

		
		self.z                = z

		self.cosmo            = cosmo
		
		closmo                = Class()
		closmo.set(self.cosmo.class_params)
		
		closmo.compute()
		
		cosmo_b             	= closmo.get_background()

		class_z             	= cosmo_b['z'][::-1]
		class_chi           	= cosmo_b['comov. dist.'][::-1]
		
		class_D 			= cosmo_b['gr.fac. D'][::-1]/cosmo_b['gr.fac. D'][-1] #normalized to todays value
		
		#for growth func
		self.w0_fld 		= -1.
		self.wa_fld 		= 0.
		
		LJ_D,zD		 	= self.get_growth(closmo,np.linspace(0.,cosmo.z_max,100))
		
		self.LJ_D_z 		= ius(zD,LJ_D)
		
		self.D_chi 			= ius(class_chi,class_D)
		self.D_z 			= ius(class_z,class_D)
		
		self.chi               = ius(class_z,class_chi)
			
		derivParams         	= closmo.get_current_derived_parameters(['z_rec'])
		
		self.z_cmb          	= derivParams['z_rec']
		
		self.chi_cmb        	= self.chi(self.z_cmb)
		
		self.a              	= self.get_a(z)

		# CLASS units: c=1, all quantities in Mpc^n
		self.H_0            	= cosmo_b['H [1/Mpc]'][-1]*const.LIGHT_SPEED
		
		class_H 			= cosmo_b['H [1/Mpc]'][::-1]*const.LIGHT_SPEED
		
		self.H 		      = ius(class_z,class_H)
        
		self.Omega_m0       	= (cosmo_b['(.)rho_cdm'][-1]+cosmo_b['(.)rho_b'][-1])/(cosmo_b['(.)rho_crit'][-1])
		
		self.prefacs       	= self.Poisson_factor()
  
  
		print closmo.get_current_derived_parameters(['Neff'])
		print closmo.get_current_derived_parameters(['h'])
#		print closmo.get_current_derived_parameters(['m_ncdm_in_eV'])
#		print closmo.get_current_derived_parameters(['m_ncdm_tot'])
		
		closmo.struct_cleanup()
		closmo.empty()
		

		## Check if chi interpolation works and if matter power spectrum makes sense
		if test:
			#should be the same, if interpolation works correctly
			pl.figure()
			pl.plot(self.z,self.chi(self.z), label="class interpolated", marker="o")
			pl.plot(class_z[::-1],class_chi[::-1], label="class", color="r", ls="--")
			pl.xlim(min(self.z),max(self.z))
			pl.xlabel("z")
			pl.ylabel("Com. Distance [Mpc]")
			pl.legend()
			pl.show()
			
			test_params=copy.deepcopy(cosmo.class_params)
			k_aux=np.exp(np.linspace(np.log(0.004*0.72),np.log(1.),200))
			test_params['z_max_pk']=10.				
			#Maximum k value in matter power spectrum
			test_params['P_k_max_1/Mpc'] = max(k_aux)
			test_params['k_min_tau0'] = min(k_aux*13000.)
			test_params['output']= 'tCl lCl mPk'
			test_params['tol_perturb_integration']=1.e-6
			#test_params['non linear']='halofit'
			
			closmo_test = Class()
			closmo_test.set(test_params)
		
			closmo_test.compute()
			
		
			test_zs=np.array([0.,0.6,1.,1.5,10.])
			P=np.empty((len(test_zs),len(k_aux)))
			for ii in range(len(test_zs)):
				P[ii]=[closmo_test.pk(k,test_zs[ii]) for k in k_aux]
			print P[ii].shape
			
			# this plot can be compared with literature
			pl.figure()
			pl.loglog(k_aux/(self.H_0/100.),np.array(P[0])*(self.H_0/100.)**3)
			pl.xlabel(r'$k[h/Mpc]$')
			pl.ylabel(r'$P[Mpc/h]^3$')
			pl.show()
			
			#check if growth function is correct
			pl.figure()
			ii=0
			for zi in test_zs:
				pl.plot(k_aux,[self.D_z(zi)**2]*len(k_aux), label=r'$D(z=%.1f)^2$'%zi, ls=":")
		
				pl.plot(k_aux,[self.LJ_D_z(zi)**2]*len(k_aux), label=r'$D(z=%.1f)^2$'%zi, ls="--")			 		 	 				
				pl.plot(k_aux,np.array(P[ii])/np.array(P[0]),label=r'$P(z=%d)/P(0)$'%zi)
				ii+=1
			pl.ylim(1e-4,1.)
			pl.xlim(min(k_aux),max(k_aux))
			pl.xlabel(r'$k[h/Mpc]$')
			pl.legend()
			pl.show()
			
			closmo_test.struct_cleanup()
			
			closmo_test.empty()
			
				
	def get_a(self,z):
		""" converts a to z """
		a=1./(1.+z)
		return a
        
	def get_z(self,a):
		""" converts z to a """
		z=(1./a)-1.
		return z
	
	def Poisson_factor(self):
		""" computes the proportionality constant of the Poisson equation """
		
		alpha= 3*self.H_0**2.*self.Omega_m0/(const.LIGHT_SPEED**2)

		return alpha
  
     	def dchidz(self,z):
		""" dDcom/dz """
		
		result = const.LIGHT_SPEED/self.H(z)

		return result
		
	def get_Cls(self,nl=False,lmax=8000):
		
		tag=self.cosmo.tag
		
		params=copy.deepcopy(self.cosmo.class_params)
		params['output']='tCl lCl mPk'
		params['lensing']='yes'
		if nl:
			params['non linear']="halofit"
			tag+="_nl"
		else:
			params['non linear']=" "
			
				
		closmo 					 = Class()
		closmo.set(params)
		
		print "Calculalating Cls... with settings",self.cosmo.class_params
		
		closmo.compute()
		
		cl_len=closmo.lensed_cl(lmax)
		
		cl_unl=closmo.raw_cl(lmax)
		
		pickle.dump([params,cl_unl,cl_len],open('/afs/mpa/home/vboehm/CosmoCodes/18_06_15/class_outputs/class_cls_%s.pkl'%tag,'w'))
			
		return True
		
	
	def get_Pm(self,kmin,kmax,k_array,test=False,nl=False,get_n=False,z_max=1.5,z_=None):
																				
		params=copy.deepcopy(self.cosmo.class_params)
          
		if params['output']!='tCl lCl mPk':
			params['output']='tCl lCl mPk'
			params['tol_perturb_integration']=1.e-6
			
   
		params['z_max_pk']     = z_max
		params['P_k_max_1/Mpc']= kmax
		if kmin>1e-5:
				params['k_min_tau0']   = kmin*13000.
		else:
				params['k_min_tau0']   = 1e-6*13000.
		print kmin, 1e-6
		
		if nl:
			params['non linear'] = "halofit"
		else:
			params['non linear'] = ""
		
		closmo 					      = Class()
		closmo.set(params)
		
		print "Calculalating matter power spectrum... with settings",self.cosmo.class_params
		
		closmo.compute()
		if nl:
			P_nl 			= np.array([closmo.pk(k,0.) for k in k_array])
			assert(False)   
		else:
			P 	 		    = np.array([closmo.pk(k,0.) for k in k_array])
		
		sigma8 		 	= closmo.sigma8()
		
		self.k_NL 				= []

		k_i=k_array#1./self.cosmo.class_params['h']
		for z in z_[np.where(z_<=z_max)]:
			Pk = np.asarray([closmo.pk(k,z) for k in k_i])   
			print z   
			self.k_NL+=[min(k_i[np.where(Pk*k_i**3/(2*np.pi**2)>1.)])]
 
#		self.k_NL=np.asarray(self.k_NL)*self.cosmo.class_params['h']
		
		print "sigma8:", sigma8
			
		self.sigma8_z 		= splrep(z_,sigma8/(self.LJ_D_z(z_))) #sigma_8 today rescaled to other redshifts
		
		if get_n:
			try:
				assert(nl==False)
			except:
				raise ValueError('Spectral index should only be calculated from linear power spectrum!')
			self.n   = HF.get_derivative(np.log(k_array),np.log(P),method="si")

		h   = self.cosmo.class_params['h']
		k_  = np.exp(np.linspace(np.log(1e-4*h),np.log(100.*h),100))
		print min(k_)
		pl.figure()
		for z_ in [0.,1.,z_max]:
				plk =[]
				for kk in k_:
						plk+=[closmo.pk(kk,z_)]
				pl.loglog(k_/h,np.asarray(plk)*h**3,label='z=%d'%z_)
		pl.xlabel(r'$k [h/Mpc]$')
		pl.xlim(1e-4,1.)    
		pl.ylabel(r'$P(k) [Mpc/h]^3$')
		pl.ylim(0.1,100000)
		pl.legend(loc='best')
		pl.savefig('pow_spec_lin.png')
   

				
		if test:
			#plots D_+/a, compare e.g. Cosmology script Matthias 
			pl.figure()
			pl.plot(self.z,self.D_z(self.z)*(1.+self.z), label="class interpolated", marker="o")
			pl.xlim(min(self.z),max(self.z))
			pl.xlabel("z")
			pl.ylabel("Growth Function D_+/a")
			pl.legend()
			pl.show()
			
			if get_n:
				
				pl.figure()
				pl.semilogx(k_array,splev(np.log(k_array),self.n,ext=2),marker="o")				
				pl.xlim(kmin,kmax)
				pl.xlabel("k")
				pl.ylabel("spectra index n")
				pl.show()
    
		closmo.struct_cleanup()
		closmo.empty()
			
	def get_abc(self,k,z,z_max):

		#checked
		a1 = 0.484
		a4 = 0.392
		a7 = 0.128
		a2 = 3.740
		a5 = 1.013
		a8 = -0.722
		a3 = -0.849
		a6 = -0.575
		a9 = -0.926
	
		try:
			self.n
		except:
			self.get_Pm(min(k),max(k),k,test=False,nl=False,get_n=True,z_max=z_max,z_=z)
	
		n=splev(np.log(k),self.n,ext=2)

		#checked
		Q=(4.-2.**n)/(1.+2**(n+1.))
	
		#k_NL for every z?
		

		self.a_nk=[]
		self.b_nk=[]
		self.c_nk=[]
		j=0
		#checked
		pl.figure()  
		for z_ in z[np.where(z<=z_max)]:
			q=k/self.k_NL[j]
			a_nk_z=(1.+splev(z_,self.sigma8_z)**a6*np.sqrt(0.7*Q)*(a1*q)**(n+a2))/(1.+(q*a1)**(n+a2))
   			pl.plot(k,a_nk_z)
			self.a_nk+=[splrep(k,a_nk_z)]
			b_nk_z=(1.+0.2*a3*(n+3.)*(q*a7)**(n+3.+a8))/(1.+(q*a7)**(n+3.5+a8))
			c_nk_z=(1.+4.5*a4/(1.5+(n+3.)**4)*(q*a5)**(n+3.+a9))/(1.+(q*a5)**(n+3.5+a9))
			pl.plot(k,b_nk_z)
			pl.plot(k,c_nk_z)
			self.b_nk+=[splrep(k,b_nk_z)]
			self.c_nk+=[splrep(k,c_nk_z)]
			j+=1
#		self.b_nk=splrep(k,b_nk)
#		self.c_nk=splrep(k,c_nk)
  
		k_test=np.exp(np.linspace(np.log(min(k)),np.log(max(k)),200))

		pl.semilogx(k_test,splev(k_test,self.b_nk[0]),ls="--")
		pl.plot(k_test,splev(k_test,self.c_nk[0]),ls="--")
		pl.plot(k_test,splev(k_test,self.a_nk[0]),ls="--")
      
		pl.savefig("bn_cn_test.png")	
#
###### Steffens growth factor version, needed if no cosmological constant, needs to be tested#########        
	def Omega_m(self,a,closmo):
		z=self.get_z(a)
		result = ( closmo.Omega_m()*(1.+z)**3 * (closmo.Hubble(0)/closmo.Hubble(z))**2 )
		return result

	def w_de(self, a, closmo):
		result = self.w0_fld + (1. - a) * self.wa_fld
		return result
    
    #defined in Linder+Jenkins MNRAS 346, 573-583 (2003)
    #solved integral by assuming linear w_de scaling analytically
	def x_plus(self,a,closmo):
		aux = 3.0 * self.wa_fld * (1. - a)
		result = closmo.Omega_m() / (1. - closmo.Omega_m()) * a**(3. * (self.w0_fld + self.wa_fld)) * np.exp(aux)
		return result
#
# growth function D/a from Linder+Jenkins MNRAS 346, 573-583 (2003)
# independent of a at early times: initial conditions g(a_early) = 1, g'(a_early) = 0
# choose a_early ~ 1./30. where Omega_m ~ 1
	def g(self,y,a,closmo):
		y0 = y[0]
		y1 = y[1]
		y2 = -(7./2. - 3./2. * self.w_de(a,closmo)/(1+self.x_plus(a,closmo))) * y1 / a - 3./2. * (1-self.w_de(a,closmo))/(1.+self.x_plus(a,closmo)) * y0 / a**2
		return y1, y2
    
	def get_growth(self,closmo,z_array):
        
#        if self.class_setup==False:
#            self.closmo = Class()
#            self.cosmo.class_params['P_k_max_1/Mpc'] = 15.
#            self.closmo.set(self.cosmo.class_params)
#    
#            self.closmo.compute()
#            self.class_setup=True
        
#        if (z_array.all()==self.z.all()):
#            z_array=self.z[np.where(self.Omega_z<0.9999)]
            
		a_array=self.get_a(z_array[::-1])
        
		init = a_array[0], 1.

		solution = odeint(self.g, init, a_array, args=(closmo,))

		growspline = a_array*solution[:,0]/solution[-1,0]

		return growspline[::-1], z_array
##### Steffens version ######### 
