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

import matplotlib.pyplot as pl

from classy import Class
import Cosmology as C
import pickle
import copy

def compute_power_spectrum(ell_min, ell_max,kmin, kmax,z,nl,bias,params):

    ell = np.exp(np.linspace(np.log(ell_min),np.log(ell_max),400))

    data    = C.CosmoData(params,z)
    chi     = data.chi(z)

    k    = np.outer(1./chi,ell+0.5)

    k_   = np.exp(np.linspace(np.log(kmin),np.log(kmax),50))

    spec_z  = np.zeros((len(z),len(ell)))

    #params['l_switch_limber']=1.#00
    params['output']='tCl, mPk'
    params['z_max_pk'] = max(z)
    params['P_k_max_1/Mpc'] = kmax
    params['k_min_tau0'] = kmin*13000.
    params['perturb_sampling_stepsize']=0.01


    if nl:
        params['non linear'] = "halofit"
    else:
        params['non linear'] = ""

    closmo=Class()
    closmo.set(params)
#
    print "Initializing CLASS..."
    print params
    closmo.compute()
    cosmo_pk = closmo.pk
#
    z_cmb    = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    print '$\sigma_8$=', closmo.get_current_derived_parameters(['sigma8'])
    chi_cmb  = data.chi(z_cmb)

    W_lens  = ((chi_cmb-chi)/(chi_cmb*chi))*(z+1.)
    kernel  = (W_lens*chi)**2


    for ii in xrange(len(z)):
        print(ii)
        spec =[cosmo_pk(k_[j],z[ii]) for j in xrange(50)]
        spec = np.array(spec)
        spec_z[ii] = np.interp(k[ii],k_,spec,left=0.,right=0.)

    spec_z= np.transpose(spec_z)

    C_pp=[]
    for ii in xrange(len(ell)):
        C_pp+=[simps(kernel*spec_z[ii],chi)]

    C_pp=np.array(C_pp)
    C_pp*=(data.prefacs**2/(ell+0.5)**4)

    dchidz  = data.dchidz(z)
    norm    = simps(dndz(z),z)
    dzdchi  = 1./dchidz

    W_gal   = dndz(z)/norm

    kernel_x  = W_gal*W_lens*bias*dzdchi
    kernel_gg = (W_gal*bias*dzdchi/chi)**2


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
    print 'bla'
    LSST        = False

    if LSST:
        dn_filename = 'dndz_LSST_i27_SN5_3y'
        red_bin     = '0'
        bounds      = {'0':[0.0,0.5],'1':[0.5,1.],'2':[1.-2.]}
    else:
        dn_filename = 'red_dis_func'
        red_bin     = 'None'
        z0=1./3.
        def red_dis(z):
            return (z/z0)**2*np.exp(-z/z0)


    #choose Cosmology (see Cosmology module)
    params      = C.SimulationCosmology

    z_min       = 1e-5

    bin_num     = 200

    tag         = params[0]['name']

    nl          = True

    ell_min     = 10
    ell_max     = 10000
    kmin        = 1e-4
    kmax        = 10

    print ell_min, ell_max

    if nl:
        tag+='_nl'

    path='/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/power_spectra/'


    class_params=copy.deepcopy(params[1])
    class_params['non linear']='halofit'
    class_params['output']='tCl, lCl'
    class_params['lensing']='yes'
    class_params['l_max_scalars']=ell_max
    class_params['l_switch_limber']=1#3000
    #class_params['perturb_sampling_stepsize']=0.01
    print 'beginning computation with ',class_params
    closmo      = Class()
    closmo.set(class_params)
    closmo.compute()
    cl_len= closmo.lensed_cl(int(ell_max))
    cl_unl= closmo.raw_cl(int(ell_max))
    pickle.dump([class_params,cl_unl,cl_len],open('../class_outputs/class_cls_%s.pkl'%tag,'w'))
    cl_phiphi       = cl_len['pp'][int(ell_min):int(ell_max)]
    ells            = cl_len['ell'][int(ell_min):int(ell_max)]


    #set up z range and binning in z space
    z_cmb       = closmo.get_current_derived_parameters(['z_rec'])['z_rec']

    zmaxs=[1.,2.,5.,z_cmb-0.01]
    clpp=[]
    for z_max in zmaxs:#1.5

      closmo.struct_cleanup()
      closmo.empty()

      z           = np.exp(np.linspace(np.log(z_min),np.log(z_max),bin_num))

      print "z_cmb: %f"%z_cmb

      if LSST:
          gz, dgn     = pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
          dndz        = interp1d(gz, dgn, kind='linear')
          z_g         = np.linspace(max(bounds[red_bin][0],z_min),bounds[red_bin][1],bin_num)
          dndz        = dndz(z_g)
          dndz        = interp1d(z_g, dndz, kind='linear',bounds_error=False,fill_value=0.)
          bias        = z+1.
      else:
          bias        = 1.
          dndz        = interp1d(z,red_dis(z),bounds_error=False,fill_value=0.)


      ll, cl_pp, cl_gg, cl_xx = compute_power_spectrum(ell_min, ell_max, kmin,kmax, z, nl,bias,copy.deepcopy(params[1]))
      clpp+=[cl_pp]

    pickle.dump([ll,clpp,zmaxs],open(path+'cl_pp_zbins_%s.pkl'%(tag),'w'))
#
#
    pl.figure()
    pl.loglog(ll,cl_pp,label='myself')
    pl.loglog(ells[1::],cl_phiphi[1::],label='Class')
    pl.legend()
    pl.show()

#    N0              = np.interp(ll,L_s,AL)
#
#    fsky            = 1.#0.5
#
#    n_bar           = np.inf#simps(dndz(z),z)*(180*60/np.pi)**2
#
#
#    noise_pp      = np.sqrt(2./(2.*ll+1.)/fsky)*(1./4.*(ll*(ll+1.))**2*(cl_pp+N0))
#    noise_gg      = np.sqrt(2./(2.*ll+1.)/fsky)*(cl_gg+1./n_bar)
#
#    noise_gp      = 1./(2.*ll+1.)/fsky
#    noise_gp*=((cl_gg+1./n_bar)*((1./2.*(ll*(ll+1.)))**2*(cl_pp+N0))+(1./2.*(ll*(ll+1))*cl_xx)**2)
#    noise_gp      = np.sqrt(noise_gp)
#
#
#    pickle.dump([ll,cl_pp+N0,cl_gg+1./n_bar,cl_xx],open('Gaussian_variances_CMB-S4_bin%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'w'))
#
#
#    pl.figure(figsize=(8,7))
#    pl.errorbar(ll, cl_gg , color='g', yerr=noise_gg, label=r'$C_L^{gg}$',elinewidth=0.1)
#    noise_gp[np.where(noise_gp>1./2.*(ll*(ll+1))*cl_xx)]=1./2.*(ll*(ll+1))*cl_xx-1e-20
#    pl.errorbar(ll,1./4.*(ll*(ll+1.))**2*cl_pp,yerr=noise_pp,label=r'$C_L^{\kappa \kappa}$',elinewidth=0.1)
#    pl.loglog(ells,1./4.*(ells*(ells+1.))**2*cl_phiphi, 'k',label=r'$C_L^{\kappa \kappa}$ theory')
#    pl.errorbar(ll,1./2.*(ll*(ll+1))*cl_xx, yerr=noise_gp, color='r',label=r'$C_L^{\kappa g}$',elinewidth=0.1)
#    pl.legend(loc='lower left',ncol=3, columnspacing=0.8, frameon=False)
#    pl.ylim([1e-9,1e-4])
#    pl.xlim([2,2000])
#    pl.xlabel('L')
#    pl.savefig('Cross_Spectra_%s.pdf'%(tag+dn_filename+red_bin))
#    pl.show()
#
#    pl.figure(figsize=(8,7))
#    pl.loglog(ll, cl_gg , color='g', label=r'$C_L^{gg}$, z=0-0.5')
#    pl.legend(loc='lower left',ncol=3, columnspacing=0.8, frameon=False)
#    pl.ylim([1e-8,1e-5])
#    pl.xlim([1,2000])
#    pl.xlabel('L')
#    pl.savefig('CLgg_%s.pdf'%(tag+dn_filename+red_bin))
#    pl.show()


#    noise_gp={}
#    for field in ['eb','tt']:
#        L_s  = AI['ls']
#        AL   = AI[field]
#        N0   = np.interp(ll,L_s,abs(AL))
#        noise_gp[field]=np.sqrt(1./(2.*ll+1.)/fsky*((cl_gg+1./n_bar)*(cl_pp+N0)+(cl_xx)**2))
#
#    pickle.dump([ll,cl_xx,noise_gp],open('cross_signal_noise_%s_%s_%s.pkl'%(red_bin,tag,dn_filename),'w'))