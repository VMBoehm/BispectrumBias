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
from scipy.interpolate import RectBivariateSpline, interp1d
import pickle

print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(-1,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model
from copy import deepcopy
from Constants import LIGHT_SPEED


class PostBorn_Bispec():

    def __init__(self,CLASSparams,k_min=1e-4,k_max=100,cross=False, dndz=None, norm=None, lmax=None, acc=4, NL=True, zmaxint=1.):
        pars = camb.CAMBparams()
        try:
            A_s=CLASSparams['A_s']
        except:
            A_s=np.exp(CLASSparams['ln10^{10}A_s'])*1e-10

        pars.set_cosmology(H0=CLASSparams['h']*100, ombh2=CLASSparams['omega_b'], omch2=CLASSparams['omega_cdm'],omk=CLASSparams['Omega_k'],num_massive_neutrinos=0, mnu=0.0, nnu=3.046)
        pars.InitPower.set_params(As=A_s,ns=CLASSparams['n_s'],  pivot_scalar=CLASSparams['k_pivot'])
        self.results= camb.get_background(pars)

        self.cross= cross
        if self.cross:
          print 'computing cross'
        self.dndz = dndz
#Get matter power spectrum interpolation objects for fiducial model
        self.kmax = k_max
        self.kmin = k_min

        self.norm = norm


        if lmax==None:
            lmax=20000
        if acc==None:
            acc=4 #(change back to 1 unless you need high accuracy - much faster)
        self.nz = 200*acc

        #integration up to zcmb
        chistar = self.results.comoving_radial_distance(zmaxint)#self.results.conformal_time(0)- model.tau_maxvis.value
        zmax    = self.results.redshift_at_comoving_radial_distance(chistar)
        print "Postborn z_max: ", zmax
        #print "Postborn z_max integration: ", zmaxint
        #self.chimaxint = self.results.comoving_radial_distance(zmaxint)
        #print('chimax integration ', self.chimaxint)
        print('chistar ', chistar)




        k_per_logint= None
        self.PK     = camb.get_matter_power_interpolator(pars, nonlinear=NL,
        hubble_units=False, k_hunit=False, kmax=self.kmax,k_per_logint=k_per_logint,
        var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zmax)

        ls          = np.hstack((np.arange(2, 400, 1),np.arange(400, 2600, 10//acc),np.arange(2650, lmax, 50//acc),np.arange(lmax,lmax+1))).astype(np.float64)
        self.ls     = ls

        self.acc    = acc
        self.lmax   = lmax

        #Get CL kappa for M_* matrix
        nchimax     = 100*acc
        chimaxs     = np.linspace(0 ,chistar, nchimax)

        cls = np.zeros((nchimax,ls.size))
        cls2= np.zeros((nchimax,ls.size))

        for i, chimax in enumerate(chimaxs[1:]):
            if self.cross:
                cl2 = self.cl_cross(chimax)
                cl  = self.cl_kappa(chimax,chistar)
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

        if self.cross:
            cchi    = cl_chi_chistar2(chis,ls, grid=True)

        Mstar = np.zeros((ls.size,ls.size))
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            #should take care of everything
            w[k>=self.kmax]=0
            w[k<=self.kmin]=0
            #w[chis>self.chimaxint]=0
            cl = np.dot(dchis*w*self.PK.P(zs, k, grid=False)/k**4*win,cchi)
            if self.cross:
                Mstar[i,:] = cl
            else:
                Mstar[i,:] = cl*l**4

        self.Mstarsp = RectBivariateSpline(ls,ls,Mstar)

        if self.cross:
            win     = (1/chis-1/chistar)/chis**2
            # bias and scale factor cancel out
            Hz      = [self.results.h_of_z(z_) for z_ in zs]
            wing    = self.dndz(zs)*Hz/chis**2 #H is in Mpc^-1 -> do not need to divide by c
            wing/=simps(self.dndz(zs),zs)
            cl      = np.zeros(ls.shape)
            w       = np.ones(chis.shape)
            cchi    = cl_chi_chistar(chis,ls, grid=True)

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
            w[k<self.kmin]=0
            w[k>=self.kmax]=0
            #take out everything higher than zmaxint in clkappa
            #w[chis>self.chimaxint]=0
            cl[i] = np.dot(dchis,w*self.PK.P(zs, k, grid=False)*win/k**4)
        if self.cross==False:
            cl*= self.ls**4
        return cl

    def cl_cross(self, chi_source):

        chi_source = np.float64(chi_source)
        chis    = np.linspace(0,chi_source,self.nz, dtype=np.float64)
        zs      = self.results.redshift_at_comoving_radial_distance(chis)
        Hz      = [self.results.h_of_z(z_) for z_ in zs]
        dchis   = (chis[2:]-chis[:-2])/2
        chis    = chis[1:-1]
        zs      = zs[1:-1]
        Hz      = Hz[1:-1]
        win     = (1/chis-1/chi_source)/chis**2
        # bias and scale factor cancel out
        wing    = self.dndz(zs)*Hz/chis**2#H is in Mpc^-1 -> do not need to divide by c
        wing/=self.norm


        cl=np.zeros(self.ls.shape)
        w = np.ones(chis.shape)
        for i, l in enumerate(self.ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<self.kmin]=0
            w[k>=self.kmax]=0
            cl[i] = np.dot(dchis,
                w*self.PK.P(zs, k, grid=False)/k**4*win*wing)
        return cl


    def bi_born(self,l1,l2,l3,gamma=1.,sym=True):
        assert(sym==True)

        cos12 = (l3**2-l1**2-l2**2)/2/l1/l2
        cos23 = (l1**2-l2**2-l3**2)/2/l2/l3
        cos31 = (l2**2-l3**2-l1**2)/2/l3/l1
        return  - 2*cos12*((l1/l2+cos12)*self.Mstarsp(l1,l2,grid=False) + (l2/l1+cos12)*self.Mstarsp(l2,l1, grid=False) )\
                - 2*cos23*((l2/l3+cos23)*self.Mstarsp(l2,l3,grid=False) + (l3/l2+cos23)*self.Mstarsp(l3,l2, grid=False) )\
                - 2*cos31*((l3/l1+cos31)*self.Mstarsp(l3,l1,grid=False) + (l1/l3+cos31)*self.Mstarsp(l1,l3 ,grid=False) )



    def bi_born_cross(self,L1,L2,L3,gamma,sym=False):

        L1L2 = (L3**2-L1**2-L2**2)/2.
        L2L3 = (L1**2-L2**2-L3**2)/2.
        L3L1 = (L2**2-L3**2-L1**2)/2.

        if sym==False:
            #L3 associated with g
            return  gamma*\
            ((L3/L1)**2*L2L3*(L1L2*self.Mstarsp(L2,L3,grid=False)+L3L1*self.Mstarsp2(L3,L2,grid=False))+\
            (L3/L2)**2*L3L1*(L1L2*self.Mstarsp(L1,L3,grid=False)+L2L3*self.Mstarsp2(L3,L1,grid=False)))
        else:
            return  gamma*\
            (((L3/L1)**2*L2L3*(L1L2*self.Mstarsp(L2,L3,grid=False)+L3L1*self.Mstarsp2(L3,L2,grid=False))+\
            (L3/L2)**2*L3L1*(L1L2*self.Mstarsp(L1,L3,grid=False)+L2L3*self.Mstarsp2(L3,L1,grid=False)))+\
            ((L2/L3)**2*L1L2*(L3L1*self.Mstarsp(L1,L2,grid=False)+L2L3*self.Mstarsp2(L2,L1,grid=False))+\
            (L2/L1)**2*L2L3*(L3L1*self.Mstarsp(L3,L2,grid=False)+L1L2*self.Mstarsp2(L2,L3,grid=False)))+\
            ((L1/L2)**2*L3L1*(L2L3*self.Mstarsp(L3,L1,grid=False)+L1L2*self.Mstarsp2(L1,L3,grid=False))+\
            (L1/L3)**2*L1L2*(L2L3*self.Mstarsp(L2,L1,grid=False)+L3L1*self.Mstarsp2(L1,L2,grid=False))))

    def cl_bi_born(self, lset,sym):

        if self.cross:
            bi   = self.bi_born_cross
        else:
            bi   = self.bi_born

        lset = lset.astype(np.float64)
        cl   = np.zeros(lset.shape[0])
        for i, (l1,l2,l3) in enumerate(lset):
            cl[i] = bi(l1,l2,l3,gamma=1,sym=sym)
        return cl


    def plot(self,cross,gamma=1, sym=True):
        if not cross:
            gamma=1.
        lsamp = np.hstack((np.arange(2, 20, 2), np.arange(25, 200, 10//self.acc), np.arange(220, 1200, 30//self.acc),
                           np.arange(1200, min(self.lmax//2,2600), 150//self.acc),np.arange(2600, self.lmax//2+1, 1000//self.acc)))

        litems =np.zeros((lsamp.size,3))
        fig, axes = plt.subplots(1,4,figsize=(12,3.6), sharey=True, sharex=True)
        l1=100
        treestyle='-.'
        res=[]
        for p,  ax in zip([0,1], axes[0:2]):
            for i,l in enumerate(lsamp):
                if p==0:
                    litems[i,:]=l,l,l
                else:
                    litems[i,:]=l,l/2.,l/2.

            testborn = self.cl_bi_born(litems,sym)*gamma

            #print testborn
            if testborn[0]>0:
                ax.loglog(lsamp, testborn, color='r',ls='-')
            else:
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
            ax.tick_params(axis='y', which='both', labelleft='on', labelright='off')

            #convert k to phi
            if not cross:
                testborn*=8./lsamp**6
                # folded
                if p==1:
                    testborn*=2.**4

            if p==1:
                res+=['folded',lsamp,testborn]
            else:
                res+=['equilateral',lsamp,testborn]
        ax.set_xticks([100,1000])
        axes[0].set_ylabel('$b^{\kappa\kappa\kappa}_{L_1 L_2 L_3}$', fontsize=18);
        fig.subplots_adjust(hspace=0)
        fig.tight_layout(h_pad=0, w_pad=0)

        if cross:
            plt.savefig('postborn_testplot_cross_no_chis.png')
            pickle.dump(res,open('postborn_phiphig.pkl','w'))
        else:
            plt.savefig('postborn_testplot.png')
            pickle.dump(res,open('postborn_phiphiphi.pkl','w'))




if __name__ == "__main__":
    params=deepcopy(C.SimulationCosmology[1])
#    H0=params['h']*100
#    Om_b=params['omega_b']/params['h']**2
#    Om_cdm=params['omega_cdm']/params['h']**2
#    Omega_m0=Om_b+Om_cdm
#    print H0, Omega_m0
#    gamma=16./(3.*Omega_m0*H0**2)*LIGHT_SPEED**2
#    sym=False
    dndz=None
    norm=None
#    sym = True

#    z       = np.exp(np.linspace(np.log(1e-4),np.log(1000),100))
#    z0      = 1./3.
#    dndz    = (z/z0)**2*np.exp(-z/z0)
#    dndz    = interp1d(z,dndz,kind='slinear',fill_value=0.,bounds_error=False)
#    norm    = simps(dndz(z),z)

    PBB=PostBorn_Bispec(C.SimulationCosmology[1],cross=False,dndz=dndz, norm=norm)
    PBB.plot(cross=False)
#    PBB=PostBorn_Bispec(C.SimulationCosmology[1],k_min=1e-3,cross=False,dndz=dndz, norm=norm)
#    PBB.plot(cross=False)
#    PBB=PostBorn_Bispec(C.SimulationCosmology[1],k_min=1e-2,cross=False,dndz=dndz, norm=norm)
#    PBB.plot(cross=False)

    #for cross in [True]:
#        PBB=PostBorn_Bispec(C.SimulationCosmology[1],cross=cross,dndz=dndz, norm=norm)
#        PBB.plot(cross,gamma,sym)
