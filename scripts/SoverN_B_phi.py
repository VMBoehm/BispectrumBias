# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:29:14 2016
SoverN_Bphi.py
Computes S/N of B_kkk in flat sky for simple Gaussian Noise
@author: VBoehm
"""

import pylab as plt
import pickle
from scipy.integrate import simps
import numpy as np
from scipy.interpolate import interp1d
import Cosmology as Cosmo



def SN_integral(bispec, clkk, N0, Ls, ls, ang, min_index, max_index, f_sky):
    # for every L interpolate over l and angle and integrate over intrepolated 2d function

    L_integrand=[]
    lmin=Ls[min_index]
    lmax=Ls[max_index]

    bin_size = len(Ls)*len(l)
    print "integrate L from ", lmin, "to ", lmax

    for i in np.arange(min_index,max_index):

        L_  = Ls[i]

        spec= bispec[i*bin_size:(i+1)*bin_size]

        integrand=[]

        for j in np.arange(i,len(ls)): #start at l=L to not double count
            l_          = ls[j]

            spec_int    = spec[j*len(ang):(j+1)*len(ang)]

            Lli         = np.sqrt(l_*l_+L_*L_-2.*l_*L_*np.cos(theta))

            index       = np.where((Lli>=l_))

            spec_int    = spec_int[index] #restrict
            ang_        = ang[index]
            Ll          = Lli[index]
            fac         = np.ones(len(Ll))

            if j==i:
                fac*=2.
                fac[np.where(np.isclose(ang_,np.pi/3.))]=6.

            integrand  += [simps(spec_int**2/fac/((clkk(L_)+N0(L_))*(clkk(l_)+N0(l_))*(clkk(Ll)+N0(Ll))),ang_)]
        #integrand = abs(np.asarray(integrand))
        L_integrand += [simps(integrand*ls[i::],ls[i::])]


    res = simps(L_integrand*Ls[min_index:max_index],Ls[min_index:max_index])

    return lmin, lmax, res


if __name__ == "__main__":
    """ settings """
    #just copy-paste bispec and ell file here
    ellfile   = 'ell_ang_full_Lmin1_Lmax8000_lmin1_lmax10000_lenL160_dlenl20_lenang160_1e-04.pkl'
    bispecfile= 'bispec_kkk_SN_full_Namikawa_Paper_Lmin1-Lmax8000-lmax10000_halofit_GM.npy'

    #paths
    ellpath   = '/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/ells/'
    specpath  = '/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/bispectra/'

    #cosmology and survey settings (for noise)

    params      = Cosmo.Pratten #make sure Cosmology is the same for bispectrum!
    tag         = params[0]['name']
    fields      = ['tt','eb','mv']
    nl          = True #use halofit in
    N0path      ='/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/N0files/'

    thetaFWHMarcmin = 3. #beam FWHM
    noiseUkArcmin   = 0.5 #eval(sys.argv[1]) #Noise level in uKarcmin
    l_max_T         = 4000
    l_max_P         = 4000
    l_min           = 50
    L_max           = 6000 #for l integration
    L_min           = 1
    TCMB            = 2.7255e6
    div             = False #divide EB by factor of 2.5 to simulate iterative delensing

    fsky            = 0.5

    if nl:
      nl_='_nl'
    else:
      nl_=''


    if l_max_T!=l_max_P:
      lmax='mixedlmax_%d_%d'%(l_max_T,l_max_P)
    else:
      lmax=str(l_max_T)

    class_file='class_cls_%s%s.pkl'%(tag,nl_)
    inputpath='/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs/ClassCls/'

    L,l,theta = pickle.load(open(ellpath+ellfile,'r'))

    bispec    = np.load(specpath+bispecfile)

    Parameter,cl_unl,cl_len=pickle.load(open(inputpath+'%s'%class_file,'r'))



    ll          = cl_unl['ell']
    cl_phiphi   = cl_len['pp'][ll]
    #convert to kappa and interpolate
    clkk        = interp1d(ll, 1/4.*(ll*(ll+1.))**2*cl_phiphi,bounds_error=False, fill_value=np.inf)

    if div:
        print 'Dividing EB by factor 2.5!'
        no_div='div25'
    else:
        no_div='nodiv'


    filename = N0path+'%s_N0_%s_%d_%d%d_%s%s.pkl'%(tag,lmax,l_min,10*noiseUkArcmin, 10*thetaFWHMarcmin,no_div,nl_)
    print N0path+'%s_N0_%s_%d_%d%d_%s%s.pkl'%(tag,lmax,l_min,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)

    plotpath = '/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/Tests/SoverN/plots/'

    Ls, NLkk = pickle.load(open(filename,'r'))

    print(min(Ls),min(ll),max(Ls),max(Ls))

    plt.figure()
    for field in ['mv']:
        N0            = interp1d(Ls,1/4.*(Ls*(Ls+1.))**2*NLkk[field],bounds_error=False, fill_value=np.inf)
        index_max     = len(L)-1
        index_min     = 0

        max_L         = []
        min_L         = []
        SN            = []

        for index_max in [40,50,60,70,80,85,90,100,110,120,140,len(L)-1]:
            minL_, maxL_, SN_ = SN_integral(bispec, clkk, N0, L,l, theta, index_min, index_max, fsky)
            max_L     +=[maxL_]
            min_L     +=[minL_]
            SN        +=[SN_*fsky/(2*np.pi**2)]

        print min_L, max_L, np.sqrt(SN)


        plt.plot(max_L, np.sqrt(np.asarray(SN)) ,marker="o",label=field)
    plt.legend()
    plt.xlabel(r'$L_{max}$')
    plt.xlim(100,2000)
    plt.ylabel("$S/N$")
    plt.grid()
    plt.savefig(plotpath+'SN_plots'+'%s_%s_%d_%d%d_%s%s.png'%(tag,lmax,l_min,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_),bbox_inches='tight')


