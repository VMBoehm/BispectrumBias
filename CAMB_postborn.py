# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:09:11 2016

@author: vboehm
adapted from https://github.com/cmbant/notebooks/blob/master/PostBorn.ipynb
"""
from __future__ import division
import sys, os
from matplotlib import pyplot as plt
import numpy as np
import Cosmology as C
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(-1,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model


class PostBorn_Bispec():
    
    def __init__(self,CLASSparams,k_min=1e-4,k_max=100,cross=False, dndz=None, norm=None, lmax=None, acc=None, NL=True):
        pars = camb.CAMBparams()
        try:
            A_s=CLASSparams['A_s']
        except:
            A_s=np.exp(CLASSparams['ln10^{10}A_s'])*1e-10
            
        pars.set_cosmology(H0=CLASSparams['h']*100, ombh2=CLASSparams['omega_b'], omch2=CLASSparams['omega_cdm'],omk=CLASSparams['Omega_k'],num_massive_neutrinos=0, mnu=0.0, nnu=3.046)
        pars.InitPower.set_params(As=A_s,ns=CLASSparams['n_s'],  pivot_scalar=CLASSparams['k_pivot'])
        self.results= camb.get_background(pars)

        self.cross= cross
        self.dndz = dndz
#Get matter power spectrum interpolation objects for fiducial model
        self.kmax = k_max
        self.kmin = k_min
        
        self.norm = norm
        

        if lmax==None:
            lmax=20000
        if acc==None:
            acc=2 #(change back to 1 unless you need high accuracy - much faster)
        self.nz = 200*acc

        chistar = self.results.conformal_time(0)- model.tau_maxvis.value #chi_cmb
        zmax    = self.results.redshift_at_comoving_radial_distance(chistar) #z_cmb

        print "Postborn z_max: ", zmax

        k_per_logint= None
        self.PK     = camb.get_matter_power_interpolator(pars, nonlinear=NL, 
        hubble_units=False, k_hunit=False, kmax=self.kmax,k_per_logint=k_per_logint,
        var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zmax)
    
        ls          = np.hstack((np.arange(2, 400, 1),np.arange(400, 2600, 10//acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)
        self.ls     = ls
        
        self.acc    = acc
        self.lmax   = lmax

        #Get cross-CL kappa for M_* matrix
        nchimax     = 100*acc
        chimaxs     = np.linspace(0 ,chistar, nchimax)
        
        cls = np.zeros((nchimax,ls.size))
        cls2= np.zeros((nchimax,ls.size))
        for i, chimax in enumerate(chimaxs[1:]):
            if self.cross:
                cl = self.cl_cross(chimax)
                cl2= self.cl_kappa(chimax,chistar)
            else:
                cl = self.cl_kappa(chimax,chistar)
            cls[i+1,:] = cl
            if self.cross:
                cls2[i+1,:] = cl2
        cls[0,:]=0
        if self.cross:
            cls2[0,:]=0
            
        cl_chi_chistar = RectBivariateSpline(chimaxs,ls,cls)
        if self.cross:
            cl_chi_chistar2 = RectBivariateSpline(chimaxs,ls,cls2)

        #Get M_*(l,l') matrix
        chis    = np.linspace(0,chistar, self.nz, dtype=np.float64)
        zs      = self.results.redshift_at_comoving_radial_distance(chis)
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = (1/chis-1/chistar)**2/chis**2
        cl      = np.zeros(ls.shape)
        w       = np.ones(chis.shape)
        cchi    = cl_chi_chistar(chis,ls, grid=True)

        Mstar = np.zeros((ls.size,ls.size))
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k>=self.kmax]=0
            w[k<=self.kmin]=0
            cl = np.dot(dchis*w*self.PK.P(zs, k, grid=False)/k**4*win,cchi)
            if self.cross:
                Mstar[i,:] = cl
            else:
                Mstar[i,:] = cl*l**4
    
        self.Mstarsp = RectBivariateSpline(ls,ls,Mstar)
    
        if self.cross:
            win     = (1/chis-1/chistar)/chis**2
            # bias and scale factor cancel out
            #Hz      = [self.results.h_of_z(z_) for z_ in zs]
            wing    = win#self.dndz(zs)/chis**2*Hz #H is in Mpc^-1 -> do not need to divide by c
            #wing/=simps(self.dndz(zs),zs)
            cl      = np.zeros(ls.shape)
            w       = np.ones(chis.shape)
            cchi    = cl_chi_chistar2(chis,ls, grid=True)
    
            Mstar = np.zeros((ls.size,ls.size))
            for i, l in enumerate(ls):
                k=(l+0.5)/chis
                w[:]=1
                w[k>=self.kmax]=0
                w[k<=self.kmin]=0
                cl = np.dot(dchis*w*self.PK.P(zs, k, grid=False)/k**4*win*wing,cchi)
                
                Mstar[i,:] = cl

        
            self.Mstarsp2 = RectBivariateSpline(ls,ls,Mstar)

    def cl_kappa(self, chi_source, chi_source2=None):
        chi_source = np.float64(chi_source)
        if chi_source2 is None: 
            chi_source2 = chi_source
        else:
            chi_source2 = np.float64(chi_source2)
        chis    = np.linspace(0,chi_source,self.nz, dtype=np.float64)
        zs      = self.results.redshift_at_comoving_radial_distance(chis)
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        win     = (1/chis-1/chi_source)*(1/chis-1/chi_source2)/chis**2
        cl      = np.zeros(self.ls.shape)
        w       = np.ones(chis.shape)
        for i, l in enumerate(self.ls):
            k =(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            cl[i] = np.dot(dchis,w*self.PK.P(zs, k, grid=False)*win/k**4)
        if self.cross==False:
            cl*= self.ls**4
        return cl
        
    def cl_cross(self, chi_source):
        
        chi_source = np.float64(chi_source)
        chis    = np.linspace(0,chi_source,self.nz, dtype=np.float64)
        zs      = self.results.redshift_at_comoving_radial_distance(chis)
        #Hz      = [self.results.h_of_z(z_) for z_ in zs]
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        #Hz      = Hz[1:-1]
        win     = (1/chis-1/chi_source)/chis**2
        # bias and scale factor cancel out
        wing    = win#self.dndz(zs)/chis**2*Hz#H is in Mpc^-1 -> do not need to divide by c
        #wing/=self.norm
        
        
        cl=np.zeros(self.ls.shape)
        w = np.ones(chis.shape)
        for i, l in enumerate(self.ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            cl[i] = np.dot(dchis,
                w*self.PK.P(zs, k, grid=False)/k**4*win*wing)
        return cl


    def bi_born(self,l1,l2,l3):
        cos12 = (l3**2-l1**2-l2**2)/2/l1/l2
        cos23 = (l1**2-l2**2-l3**2)/2/l2/l3
        cos31 = (l2**2-l3**2-l1**2)/2/l3/l1
        return  - 2*cos12*((l1/l2+cos12)*self.Mstarsp(l1,l2,grid=False) + (l2/l1+cos12)*self.Mstarsp(l2,l1, grid=False) )\
                - 2*cos23*((l2/l3+cos23)*self.Mstarsp(l2,l3,grid=False) + (l3/l2+cos23)*self.Mstarsp(l3,l2, grid=False) )\
                - 2*cos31*((l3/l1+cos31)*self.Mstarsp(l3,l1,grid=False) + (l1/l3+cos31)*self.Mstarsp(l1,l3 ,grid=False) ) 
    
    def bi_born_cross(self,L1,L2,L3):#,gamma):
        L1L2 = (L3**2-L1**2-L2**2)/2.
        L2L3 = (L1**2-L2**2-L3**2)/2.
        L3L1 = (L2**2-L3**2-L1**2)/2.
        #return  gamma*\
        #((L3/L1)**2*L2L3*(L1L2*self.Mstarsp(L2,L3,grid=False)+L3L1*self.Mstarsp2(L3,L2,grid=False))+\
        #(L3/L2)**2*L3L1*(L1L2*self.Mstarsp(L1,L3,grid=False)+L2L3*self.Mstarsp2(L3,L1,grid=False)))
        return 16*((1./L1)**2*L2L3*(L1L2*self.Mstarsp(L2,L3,grid=False)+L3L1*self.Mstarsp2(L3,L2,grid=False))+\
        (1./L2)**2*L3L1*(L1L2*self.Mstarsp(L1,L3,grid=False)+L2L3*self.Mstarsp2(L3,L1,grid=False))+\
        (1./L3)**2*L1L2*(L3L1*self.Mstarsp(L1,L2,grid=False)+L2L3*self.Mstarsp2(L2,L1,grid=False)))*2
        
        
    def cl_bi_born(self, lset):
        
        if self.cross:
            print 'here'
            bi   = self.bi_born_cross
        else:
            print 'wrong'
            bi   = self.bi_born
            
            
        lset = lset.astype(np.float64)
        cl   = np.zeros(lset.shape[0])
        for i, (l1,l2,l3) in enumerate(lset):
            cl[i] = bi(l1,l2,l3)
        return cl
        
    
    def plot(self):

        lsamp = np.hstack((np.arange(2, 20, 2), np.arange(25, 200, 10//self.acc), np.arange(220, 1200, 30//self.acc),
                           np.arange(1300, min(self.lmax//2,2600), 150//self.acc),np.arange(3000, self.lmax//2+1, 1000//self.acc)))
        
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
        plt.savefig('postborn_testplot.png')
        
        

if __name__ == "__main__":
    
    PBB=PostBorn_Bispec(C.SimulationCosmology[1],cross=True)
    PBB.plot()
