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
from scipy.interpolate import interp1d, splev
import pickle
from copy import deepcopy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

from classy import Class
import Cosmology as C
from N32biasIntegrals import I0, I2
import CAMB_postborn as postborn
from Constants import LIGHT_SPEED

from Bispectra import Bispectra



"""---- Choose you settings here ---"""
if __name__ == "__main__":

    "---begin settings---"

    tag         = 'laptop_test1'

    #type of bispectrum
    kkg         = False
    kgg         = False
    kkk         = True

    #triangle configuration
    ell_type    ='full'#'equilat','folded'

    #compute beta integrals?
    integrals   = True

    #use LSST like redshift bins
    LSST        = False

    #if kkg or kgg: symmetrize over positions of k/g
    sym         = False

    #Limber approximation, if true set class_params['l_switch_limber']=100, else 1
    Limber      = True

    #post Born (use post Born terms from Pratten & Lewis arXiv:1605.05662)
    post_born   = False

    #fitting formula (use B_delta fitting formula from Gil-Marin et al. arXiv:1111.4477
    B_fit       = False
    fit_z_max   = 1.5

    #number of redshift bins
    bin_num     = 100
    z_min       = 1e-4

    #sampling in L/l and angle
    len_L       = 80
    len_l       = 200
    len_ang     = 200

    #ell range (for L and l)
    L_min       = 100.
    L_max       = 3000.

    l_min       = 1
    l_max       = 8000.

    k_min       = None#1e-4
    k_max       = None#100.


    Delta_theta = 1e-4

    nl          = True

    cparams     = C.SimulationCosmology#Planck2015_TTlowPlensing

    #path, where to store results
    path        = "/home/nessa/Documents/Projects/LensingBispectrum/CMB-nonlinear/outputs"


    if nl==False:
        spectrum_config='_linPs'
    else:
        spectrum_config='_lnPs'

    if ell_type=='equilat':
        len_side= 250
    if ell_type=='folded':
        len_side= 250


    "---end settings---"

    assert(kgg+kkk+kkg==1)
    assert(ell_type in ['folded','all','equilat'])


    params  = deepcopy(cparams)
    closmo  = Class()
    closmo.set(params)
    closmo.compute()
    z_cmb   = closmo.get_current_derived_parameters(['z_rec'])['z_rec']
    closmo.struct_cleanup()
    closmo.empty()
    del closmo

    print "z_cmb: %f"%z_cmb

    z       = np.exp(np.linspace(np.log(z_min),np.log(z_cmb),bin_num))

    # avoid numerical inaccuracies
    if z[-1]!=z_cmb:
        z[-1]=z_cmb

    if kkg or kgg:

        for red_bin in ['0']:

            if LSST:

                dn_filename = 'dndz_LSST_i27_SN5_3y'

                if red_bin!='None':
                    zs, bins= pickle.load(open(dn_filename+'_extrapolated.pkl','r'))
                    mbin    = bins[int(red_bin)]
                    dndz    = interp1d(zs, mbin[1], kind='linear',bounds_error=False,fill_value=0.)
                    norm    = mbin[2]
                    print 'using z-bin', mbin
                    print 'norm: ', norm
                else:
                    gz, dgn = pickle.load(open(dn_filename+'tot_extrapolated.pkl','r'))
                    dndz    = interp1d(gz, dgn, kind='linear',bounds_error=False,fill_value=0.)
                    norm    = simps(dndz(z),z)
                    z       = np.linspace(min(mbin[0][0]-0.3,z_min),mbin[0][1]+0.3,bin_num)
                bias    = z+1.

            else:
                z0      = 1./3.
                dndz    = (z/z0)**2*np.exp(-z/z0)
                dndz    = interp1d(z,dndz,kind='slinear',fill_value=0.,bounds_error=False)
                bias    = 1.
                norm    = simps(dndz(z),z)

    else:
        dndz    = None
        norm    = None
        bias    = None

        #cosmo dependent functions
        data    = C.CosmoData(params,z)


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
                L         = np.exp(np.linspace(np.log(L_min),np.log(L_max),len_L))
#                la        = np.arange(l_min,20)
#                lb        = np.exp(np.linspace(np.log(20),np.log(l_max),len_l-19))
#                l         = np.append(la,lb)
#            elif ell_type=="log_halfang":
                la        = np.linspace(l_min,20,20,endpoint=False)
                lb        = np.linspace(20,600,200,endpoint=False)
                # insufficient sampling at high l
                lc        = np.linspace(600,L_max+150,500,endpoint=False)
                ld        = np.exp(np.linspace(np.log(L_max+150),np.log(l_max),40))
                l1        = np.append(la,lb)
                l2        = np.append(lc,ld)
                l         = np.append(l1,l2)
                len_l     = len(l)
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
            theta   = np.exp(np.linspace(np.log(Delta_theta),np.log(np.pi-Delta_theta), len_ang))
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
                    for k in range(len_ang):
                        for j in range(len_l):
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

            #pickle.dump([ang12,ang23,ang31,angmu],open(filename_ang, 'w'))
            #pickle.dump(ell,open(filename, 'w'))
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

            Bi_cum=bi_cum(bs.bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, fullsky=False)

#            for R in rad:
#                Bi_cum+=[skew(bs.bi_phi, bs.ell, angmu ,len_L, len_l, len_ang, R=R,fullsky=False)]

            L    = np.unique(ell[0::3])

            if sym:
                config+='_sym'

            pickle.dump([params,Limber,L,Int0,Int2],open('./cross_integrals/I0I1I2%s.pkl'%(config),'w'))

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










