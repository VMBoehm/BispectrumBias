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
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.ioff()

from classy import Class
import Cosmology as C
import os
import pickle

        
def compute_power_spectrum(ell_min, ell_max,z,z_g,Limber,nl):

    ell = np.exp(np.linspace(np.log(ell_min),np.log(ell_max),200))
    
    cosmo   = C.Cosmology(zmin=0.00, zmax=1200, Params=params, Limber = Limber, lmax=ell_max, mPk=False, Neutrinos=False)
    data    = C.CosmoData(cosmo,z)
    chi     = data.chi(z)

    k    = np.outer(1./chi,ell+0.5)
    cosmo.class_params['l_switch_limber']=1.#00
    cosmo.class_params['perturb_sampling_stepsize']=0.01


    cosmo.class_params['output']='tCl, mPk'
    #cosmo.class_params['lensing']='yes'
    cosmo.class_params['tol_perturb_integration']=1.e-6
								
    cosmo.class_params['z_max_pk'] = max(z)		
    cosmo.class_params['P_k_max_1/Mpc'] = max(k.flatten())
    cosmo.class_params['k_min_tau0'] = min(k.flatten())*13000.
    
    if nl:
        cosmo.class_params['non linear'] = "halofit"
    else:
        cosmo.class_params['non linear'] = ""
        
    closmo=Class()
    closmo.set(cosmo.class_params)

    print "Initializing CLASS..."
    print cosmo.class_params
    closmo.compute()
    cosmo_pk = closmo.pk    
#    cl_unl   = closmo.raw_cl(ell_max)
#    cl_len   = closmo.lensed_cl(ell_max)
    
    #pickle.dump([cosmo.class_params,cl_unl,cl_len],open('../class_outputs/class_cls_%s.pkl'%tag,'w'))
    z_cmb    = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    print '$\sigma_8$=', closmo.get_current_derived_parameters(['sigma8'])
    chi_cmb  = data.chi(z_cmb)
        
    W_lens  = ((chi_cmb-chi)/(chi_cmb*chi))*(1./data.a)
    kernel  = (W_lens*chi)**2
              
    spec_z  = np.zeros((len(z),len(ell)))
    print min(k.flatten()), max(z), max(chi)
    for ii in xrange(len(z)):
        spec=[cosmo_pk(k[ii][j],z[ii]) for j in xrange(len(k[ii]))]
        spec= np.array(spec)
        spec_z[ii] = spec
    spec_z=np.transpose(spec_z)
    C_pp=[]
    for ii in xrange(len(ell)):
        C_pp+=[simps(kernel*spec_z[ii],chi)]
    C_pp=np.array(C_pp)
    C_pp*=(data.prefacs**2/(ell+0.5)**4)
    
    data    = C.CosmoData(cosmo,z_g)
    chi     = data.chi(z_g)

    k    = np.outer(1./chi,ell+0.5)

        
    W_lens  = ((chi_cmb-chi)/(chi_cmb*chi))*(z_g+1.)
    dchidz  = data.dchidz(z_g)
    norm    = simps(dndz(z_g),z_g)
    dzdchi  = 1./dchidz
    W_gal   = dndz(z_g)/norm
    
    pl.figure()
    pl.plot(z_g,W_gal)
    pl.savefig('W_gal_z_test.png')

    pl.figure()
    pl.plot(z_g,W_lens)
    pl.savefig('W_lens_z_test.png')
    
    kernel_x  = W_gal*W_lens*(z_g+1.)*dzdchi
    kernel_gg = (W_gal*(z_g+1.)*dzdchi/chi)**2
    
    pl.figure()
    pl.plot(z_g,kernel_gg)
    pl.plot(z_g,kernel_x)
    pl.savefig('kernel_gal_test.png')
    
        
    spec_z  =np.zeros((len(z_g),len(ell)))

    for ii in xrange(len(z_g)):
        spec =[cosmo_pk(k[ii][j],z_g[ii]) for j in xrange(len(k[ii]))]
        spec = np.array(spec)
        spec_z[ii] = spec
    spec_z=np.transpose(spec_z)
    cross = []
    C_gg  = []
    for ii in xrange(len(ell)):
        cross+=[simps(kernel_x*spec_z[ii],chi)]
        C_gg+=[simps(kernel_gg*spec_z[ii],chi)]
    cross = np.array(cross)
    cross*=(data.prefacs/(ell+0.5)**2)
    C_gg  = np.array(C_gg) 
    
    closmo.struct_cleanup()
    closmo.empty()
        
    return ell, C_pp, C_gg, cross
    



"""---- Choose you settings here ---"""
if __name__ == "__main__":  
    
    "---begin settings---"
 
    dn_filename = 'dndz_LSST_i27_SN5_3y'
    
    #choose Cosmology (see Cosmology module)
    params      = C.Namikawa#Planck2015_TTlowPlensing
    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = False
 
    #binbounds
    red_bin     = '0'
    
    bounds      = {'0':[0.0,0.5],'1':[0.5,1.],'2':[1.-2.]}
				    
    #number of redshift bins 
    bin_num     = 200

    #ell range (for L and l)
    ell_min     = 1
    ell_max     = 3000
      
    nl          = True

    gz, dgn     = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))

    dndz        = interp1d(gz, dgn, kind='linear')        
    #initialize cosmology
#    cosmo   = C.Cosmology(zmin=0.00, zmax=1200, Params=params, Limber = Limber, lmax=ell_max, mPk=False, Neutrinos=False)
#    closmo  = Class()
#    params[1]['l_max_scalars']=4000
#    closmo.set(params[1])
#    closmo.compute()
#    #set up z range and binning in z space
    z_min   = 1e-3
#    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
#    closmo.struct_cleanup()
#    closmo.empty()
#    
#    print "z_cmb: %f"%z_cmb
#
#    #linear sampling in z is ok
#    z       = np.exp(np.linspace(np.log(z_min),np.log(z_cmb-0.01),bin_num))
#
#
    z_g     = np.linspace(max(bounds[red_bin][0],z_min),bounds[red_bin][1],bin_num)

    tag     = params[0]['name']
    if nl:
        tag+='_nl'
    try:
        ll, cl_pp, cl_gg, cl_xx = pickle.load(open('cross_spectrum_%s_%s_bin%s.pkl'%(tag,dn_filename,red_bin),'r'))
    except:
        print 'cross_spectrum_%s_%s_bin%s.pkl not found'%(tag,dn_filename,red_bin)  
        ll, cl_pp, cl_gg, cl_xx = compute_power_spectrum(ell_min, ell_max, z, z_g, Limber, nl)
        pickle.dump([ll,cl_pp, cl_gg, cl_xx],open('cross_spectrum_%s_%s_bin%s.pkl'%(tag,dn_filename,red_bin),'w'))
    
    Parameter,cl_unl,cl_len=pickle.load(open('../class_outputs/class_cls_%s.pkl'%tag,'r'))
    cl_phiphi     = cl_len['pp'][ell_min:ell_max+1]
    ells          = cl_len['ell'][ell_min:ell_max+1]

#    noiseUkArcmin   = 1.
#    thetaFWHMarcmin = 1.
    fsky            = 0.5   
    
    n_bar           = simps(dndz(z_g),z_g)*(180*60/np.pi)**2
    
    AI            = pickle.load(open('/home/traveller/Documents/Projekte/LensingBispectrum/CosmoCodes/N0files/Namikawa_N0_mixedlmax_730.pkl','r'))
    L_s           = AI['ls']
    AL            = AI['MV']
    N0            = np.interp(ll,L_s,AL)#(ll)
    print L_s
    
    noise_pp      = np.sqrt(2./(2.*ll+1.)/fsky)*(1./4.*(ll*(ll+1.))**2*(cl_pp+N0))   
    noise_gg      = np.sqrt(2./(2.*ll+1.)/fsky)*(cl_gg+1./n_bar)
    
    noise_gp      = np.sqrt(2./(2.*ll+1.)/fsky*((cl_gg+1./n_bar)*((1./4.*(ll*(ll+1)))**2*(cl_pp+N0))
    +(1./2.*(ll*(ll+1))*cl_xx)**2))
    
    pickle.dump([ll,cl_pp+N0,cl_gg+1./n_bar,cl_xx],open('Gaussian_variances_CMB-S4_LSST_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'w'))

    pl.figure(figsize=(8,7))
    pl.errorbar(ll, cl_gg , color='g', yerr=noise_gg, label=r'$C_L^{gg}$, z=0-0.5')
    noise_gp[np.where(noise_gp>1./2.*(ll*(ll+1))*cl_xx)]=1./2.*(ll*(ll+1))*cl_xx-1e-20
    pl.errorbar(ll,1./4.*(ll*(ll+1.))**2*cl_pp,yerr=noise_pp,label=r'$C_L^{\kappa \kappa}$')
    pl.loglog(ells,1./4.*(ells*(ells+1.))**2*cl_phiphi, 'k',label=r'$C_L^{\kappa \kappa}$ theory')
    pl.errorbar(ll,1./2.*(ll*(ll+1))*cl_xx, yerr=noise_gp, color='r',label=r'$C_L^{\kappa g}$, z=0-0.5')
    pl.legend(loc='lower left',ncol=3, columnspacing=0.8, frameon=False)
    pl.ylim([1e-9,1e-4])
    pl.xlim([2,2000])
    pl.xlabel('L')
    pl.savefig('Cross_Spectra_%s.pdf'%(tag+dn_filename+red_bin))
    pl.show()
