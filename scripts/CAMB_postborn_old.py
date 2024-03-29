# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:09:11 2016

@author: vboehm
adapted from https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb
"""

import sys, os
from matplotlib import pyplot as plt
import numpy as np
import Cosmology as C
#from classy import Class
#from scipy.interpolate import splev, splrep
from scipy.interpolate import RectBivariateSpline

print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(-1,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower

#def setPlotStyle():
#    params = {'backend': 'pdf',
#              'axes.labelsize': 13,
#              'font.size': 11,
#              'legend.fontsize': 11,
#              'xtick.labelsize': 12,
#              'ytick.labelsize': 12,
#              'ytick.major.pad': 4,
#              'xtick.major.pad': 6,
#              'text.usetex': False}
#    rcParams.update(params)

class PostBorn_Bispec():

    def __init__(self,CLASSparams,k_min,k_max,lmax=None, acc=None, NL=True):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=CLASSparams['h']*100, ombh2=CLASSparams['omega_b'], omch2=CLASSparams['omega_cdm'],omk=CLASSparams['Omega_k'],num_massive_neutrinos=0, mnu=0.0, nnu=3.046)
        pars.InitPower.set_params(As=CLASSparams['A_s'],ns=CLASSparams['n_s'],  pivot_scalar=CLASSparams['k_pivot'])
        self.results= camb.get_background(pars)


#Get matter power spectrum interpolation objects for fiducial model
        self.kmax=k_max
        self.kmin=k_min

        if lmax==None:
            lmax=20000
        if acc==None:
            acc=2 #(change back to 1 unless you need high accuracy - much faster)
        self.nz = 200*acc
#        nz_bi=200*acc
#
#        ztag = 'star'
        chistar = self.results.conformal_time(0)- model.tau_maxvis.value
        zmax = self.results.redshift_at_comoving_radial_distance(chistar)

        print "Postborn z_max: ", zmax

        k_per_logint = None
        self.PK = camb.get_matter_power_interpolator(pars, nonlinear=NL,
        hubble_units=False, k_hunit=False, kmax=self.kmax,k_per_logint=k_per_logint,
        var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zmax)

#        lsall = np.arange(2,lmax+1, dtype=np.float64)
        ls = np.hstack((np.arange(2, 400, 1),np.arange(401, 2600, 10//acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)
        print len(ls)
        print len(np.unique(ls))
        self.ls=ls
        self.acc=acc
        self.lmax=lmax

        #Get cross-CL kappa for M_* matrix
        nchimax = 100*acc
        chimaxs=np.linspace(0 ,chistar, nchimax)
        cls = np.zeros((nchimax,ls.size))
        for i, chimax in enumerate(chimaxs[1:]):
            cl = self.cl_kappa(chimax,chistar)
            cls[i+1,:] = cl
            #if i%4==0: plt.semilogx(ls,cl)
        cls[0,:]=0

        cl_chi_chistar = RectBivariateSpline(chimaxs,ls,cls)

        #Get M_*(l,l') matrix
        chis =np.linspace(0,chistar, self.nz, dtype=np.float64)
        zs=self.results.redshift_at_comoving_radial_distance(chis)
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1] #not necessary for me
        zs = zs[1:-1]#not necessary for me
        win = (1/chis-1/chistar)**2/chis**2
        cl=np.zeros(ls.shape)
        w = np.ones(chis.shape)
        cchi = cl_chi_chistar(chis,ls, grid=True)

        Mstar = np.zeros((ls.size,ls.size))
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k>=self.kmax]=0
            w[k<=self.kmin]=0
            cl = np.dot(dchis*w*self.PK.P(zs, k, grid=False)*win/k**4,cchi)
            Mstar[i,:] = cl*l**4 #(l*(l+1))**2

        self.Mstarsp = RectBivariateSpline(ls,ls,Mstar)

    def cl_kappa(self, chi_source, chi_source2=None):
        chi_source = np.float64(chi_source)
        if chi_source2 is None:
            chi_source2 = chi_source
        else:
            chi_source2 = np.float64(chi_source2)
        chis = np.linspace(0,chi_source,self.nz, dtype=np.float64)
        zs=self.results.redshift_at_comoving_radial_distance(chis)
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]
        win = (1/chis-1/chi_source)*(1/chis-1/chi_source2)/chis**2
        cl=np.zeros(self.ls.shape)
        w = np.ones(chis.shape)
        for i, l in enumerate(self.ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            cl[i] = np.dot(dchis,
                w*self.PK.P(zs, k, grid=False)*win/k**4)
        cl*= self.ls**4 #(ls*(ls+1))**2
        return cl

    def bi_born(self,l1,l2,l3):
        cos12 = (l3**2-l1**2-l2**2)/2/l1/l2
        cos23 = (l1**2-l2**2-l3**2)/2/l2/l3
        cos31 = (l2**2-l3**2-l1**2)/2/l3/l1
        return  - 2*cos12*((l1/l2+cos12)*self.Mstarsp(l1,l2,grid=False) + (l2/l1+cos12)*self.Mstarsp(l2,l1, grid=False) )\
                - 2*cos23*((l2/l3+cos23)*self.Mstarsp(l2,l3,grid=False) + (l3/l2+cos23)*self.Mstarsp(l3,l2, grid=False) )\
                - 2*cos31*((l3/l1+cos31)*self.Mstarsp(l3,l1,grid=False) + (l1/l3+cos31)*self.Mstarsp(l1,l3 ,grid=False) )

    def cl_bi_born(self, lset):
        bi=self.bi_born
        lset = lset.astype(np.float64)
        cl=np.zeros(lset.shape[0])
        for i, (l1,l2,l3) in enumerate(lset):
            cl[i] = bi(l1,l2,l3)
        return cl

    def plot(self,ll=None):

#        if np.all(ll!=None):
        lsamp = np.hstack((np.arange(2, 20, 2), np.arange(25, 200, 10//self.acc), np.arange(220, 1200, 30//self.acc),np.arange(1300, min(self.lmax//2,2600), 150//self.acc),np.arange(3000, self.lmax//2+1, 1000//self.acc)))
#        else:
#            lsamp = ll
#
        litems =np.zeros((lsamp.size,3))
        fig, axes = plt.subplots(1,4,figsize=(12,3.6), sharey=True, sharex=True)
        l1=100
        treestyle='-.'
        for p,  ax in zip([0,1], axes[0:2]):
            for i,l in enumerate(lsamp):
                if p==0:
                    litems[i,:]=l,l,l
                else:
                    litems[i,:]=l,l/2.,l/2.

            testborn = self.cl_bi_born(litems)

            ax.loglog(lsamp, testborn, color='r',ls='-')
            ax.loglog(lsamp, -testborn, color='r',ls='--',label='_nolegend_')

            ax.legend(['Post-Born'], frameon =False, loc='lower left')
            if p==1:
                ax.set_title('$L_1, L_2, L_3 = L, L/2, L/2$')
                ax.text(1800,1e-14,'Folded')
            else:
                ax.set_title('$L_1,L_2, L_3 = L, L, L$')
                ax.text(1800,1e-14,'Equilateral')

            ax.set_xlabel('$L$')
            ax.set_xlim([l1,1e4])
            ax.set_ylim([1e-20,1e-13])


        ax.set_xticks([100,1000])
        axes[0].set_ylabel('$b^{\kappa\kappa\kappa}_{L_1 L_2 L_3}$', fontsize=18);
        fig.subplots_adjust(hspace=0)
        fig.tight_layout(h_pad=0, w_pad=0)
        plt.show()
        plt.savefig('postborn_testplot.png')



if __name__ == "__main__":

    PBB=PostBorn_Bispec(C.SimulationCosmology[1],1e-3,50)
    PBB.plot()
