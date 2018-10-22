#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:12:16 2018

@author: nessa
"""

from __future__ import division
import numpy as np
import pickle
import Cosmology as Cosmo
import matplotlib.pyplot as plt

params  = Cosmo.SimulationCosmology
tag     = params[0]['name']
field   = 'tt'
nl      = True
div     = False

thetaFWHMarcmin = 1.
noiseUkArcmin   = 1.
lmin            = 500
l_max_T         = 4000
l_max_P         = 4000
len_ang         = 800
len_l           = 5040
bs_tag          = 'full'


biaspath='./biasResults/lmin%d_noise%d_theta%d/%s/'%(lmin,noiseUkArcmin,thetaFWHMarcmin*10,bs_tag)
ALpath  ='./outputs/N0files/'

outputpath='/home/nessa/Documents/Projects/LensingBispectrum/Simulations/NewRuns/CMBLensSims/biastheory/'


if div:
    no_div='div25'
else:
    no_div='nodiv'

if l_max_T!=l_max_P:
    lmax='mixedlmax'
else:
    lmax=str(l_max_T)


if nl:
  nl_='_nl'
else:
  nl_=''

class_file='class_cls_%s%s.pkl'%(tag,nl_)

inputpath='./outputs/ClassCls/'

Ls, bias = pickle.load(open(biaspath+'Totbias_%d_%d.pkl'%(len_ang,len_l)))
print(bias)
A_L_file  = ALpath+'%s_N0_%s_%s_%d%d_%s%s.pkl'%(tag,lmax,lmin,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)

LA, NL_KK = pickle.load(open(A_L_file,'r'))


AL        = np.interp(Ls,LA,NL_KK['tt'])

class_params,cl_unl,cl_len = pickle.load(open(inputpath+class_file,'r'))
clpp  = cl_len['pp']
ll    = cl_len['ell']
cltt  = cl_len['tt']
cltt_unl = cl_unl['tt']

clphiphi =np.interp(Ls,ll,clpp)

print(Ls)
#pickle.dump([Ls,2.*AL*bias],open(outputpath+'N32bias_%s_%d_%d_%d.pkl'%(bs_tag,thetaFWHMarcmin,noiseUkArcmin,lmin),'w'))


