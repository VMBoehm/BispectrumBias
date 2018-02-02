"""
Created on Thu Mar 10 15:53:46 2016

@author: vboehm
"""

from __future__ import division
import numpy as np
import os
import pickle
from TotBias import get_bias
from mpi4py import MPI
from scipy.integrate import simps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


### ----------------  settings ------------------###
cosmo_tag   ="Jias_Simulation_new_nl"
field       ='TT'
res_tag     ="kkk_fullanalytic_red_dis_lnPs_Bfit_Jias_Simulationsim_comp_1_Lmin1-Lmax2999-lmax8000-lenBi1200000"

bispec_file ='../interp/bispec_interp_%s.pkl'%res_tag

thetaFWHMarcmin = 1.
noiseUkArcmin   = 1.
lcutmin         = 500
lcutmax         = 4000
TCMB            = 2.725e6
jjs             = [20,30]

N = 1000 #len(mu1s)
M = 1000 #len(mus)
K = 2**5#**12#len(ls)=4096=2**12 #max ls= len(ls)+minl
P = 2**5#**12 #len(l1s) #max l1s= len(l1s)+minl1 #dividable 256 to distribute among cores
### ----------------  settings ------------------###

cl, nl = {}, {}

res_path='/global/homes/v/vboehm/N32theory/biasResults/'

try:
    os.stat(res_path+'lmin%d'%lcutmin)
except:
    os.mkdir(res_path+'lmin%d'%lcutmin)

##load power spectra created with class (Parameter is a list of the cosmologocal parameters used)
Parameter,cl_unl,cl_len=pickle.load(open('/global/homes/v/vboehm/N32theory/ClassCls/class_cls_%s.pkl'%cosmo_tag,'r'))

##load spline interpolation of bispectrum

## bispectrum was interpolated along mu-axis for l=l3 and L kept fixed
# x: mu for which it was interpolated
# Ls, l3s: ls and ls for which it was intrepolated
# spline is a 2D array of splines: splines[i][j] interpolates B(L[i],l[j],mu) in mu
ang_i, Ls ,ls, splines = pickle.load(open(bispec_file))

ang_min=min(ang_i)
ang_max=max(ang_i)

cl['TT']=cl_len['tt']
cl['TT_unlen']=cl_unl['tt']
cl_phiphi=cl_len['pp']


lmax=max(cl_unl['ell'])
lmin=min(cl_unl['ell'])

assert(lmin==0)
assert(lmax>=4000)
ells=np.arange(lmin,lmax+1)

### beam, noise
thetaFWHM   = thetaFWHMarcmin*np.pi/(180.*60.)
deltaT      = noiseUkArcmin/thetaFWHMarcmin
ll          = ells
nlI         = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2

nlI[0:lcutmin]   = 1e20
nlI[lcutmax+1::] = 1e20
nl['TT']         = nlI

for j in jjs:
	if rank==0:
		print j, Ls[j]
	spline_cut=splines[j*len(ls):(j+1)*len(ls)]
	num, L, l1, res, M, P= get_bias(Ls[j],ls,len(ls),cl,field,nl,lmin,lmax,ang_min,ang_max, j,rank,size,splines,N,M,K,P)

	res_all=comm.gather(res,root=0)
	l1s=comm.gather(l1,root=0)
	if rank==0:
		#combine to one array
		integrand=np.concatenate(res_all,axis=0)
		x=np.concatenate(l1s,axis=0)
		#do integration over l1
		sumres=simps(integrand,x)
		#save final result
		pickle.dump([L,sumres],open(res_path+'lmin%d/A1C1_%d_%d_%d_%s.pkl'%(lcutmin,num,M,P,res_tag),'w'))


