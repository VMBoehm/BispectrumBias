from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classy import Class


#example cosmology
Planck2015=[{
'name':"Planck2015"},{
'h': 0.68,
'omega_b' : 0.0223,
'omega_cdm': 0.119,
'Omega_k' : 0.0,
'ln10^{10}A_s':3.062,#sigma8:  0.829472350106
'n_s'    : 0.97,
'k_pivot' : 0.05}]


"""------------settings----------------"""
params      = Planck2015[1] #plug in your CLASS parameter dictionary here
tag         = Planck2015[0]['name'] #give it a nametag
fields      = ['tt','te','ee','eb','bb','tb']
nl          = True #use non-linear Cl_phiphi
out_path    = './'#outputpath for results

thetaFWHMarcmin = 1. #beam FWHM
noiseUkArcmin   = 1. #Noise level in uKarcmin
l_max_T         = 3000
l_max_P         = 5000
l_min           = 50
L_max           = 6000 #for l integration
L_min           = 1
TCMB            = 2.7255e6
div             = True #divide EB by factor of 2.5 to imitate iterative estimator

if nl:
  nl_='_nl'
else:
  nl_=''

class_file='class_cls_%s%s.pkl'%(tag,nl_)
inputpath='./outputs/ClassCls/'

"""------------settings----------------"""

def get_del_lnCl_del_lnl(lvec, Cl, test_plots=False):
#for large lens limit. did not look at this in while, might not be compatible with rest of the code anymore,
    lnl = np.log(np.array(lvec, dtype=float))
    lnCl = np.log(Cl)
    approx_del_lnCl_del_lnl = np.diff(lnCl)/np.diff(lnl)
    del_lnCl_del_lnl = np.zeros(lvec.shape[0])
    # central difference
    del_lnCl_del_lnl[1:-1] = (lnCl[2:] - lnCl[:-2]) / ( lnl[2:] - lnl[:-2])
    # forward difference
    if lnl[1]-lnl[0] == 0.:
        del_lnCl_del_lnl[0] = np.nan
    else:
        del_lnCl_del_lnl[0] = (lnCl[1]-lnCl[0]) / (lnl[1]-lnl[0])
    # backward difference
    del_lnCl_del_lnl[-1] = (lnCl[-1]-lnCl[-2]) / (lnl[-1]-lnl[-2])

    if test_plots:
        plt.semilogx(lvec, lnCl, label='ln Cl')
        plt.semilogx(lvec[1:], approx_del_lnCl_del_lnl, label='approx deriv')
        plt.semilogx(lvec, del_lnCl_del_lnl, label='deriv')
        plt.legend()
        plt.show()

    return del_lnCl_del_lnl

def get_lowL_limit(C_unlen, C_len, C_tot, ell,L):
    """ could enforce integration limits..."""
    #large lens limit. did not look at this in while, might not be compatible with rest of the code anymore,


    del_lnCl_del_lnl=get_del_lnCl_del_lnl(ell,C_unlen)
    D = 1. +del_lnCl_del_lnl+3./8.*del_lnCl_del_lnl**2

    integral=np.sum((C_len/C_tot)**2.*D*(2*ell+1))**-1

    result= integral*8*np.pi/((L)*L)**2

    return result


def get_cl2(cl_unlen, cl_tot, l2, lmin, lmax):

	## If l2 lies in range of known Cl's, linearly interpolate to get cl2
	cl2_unlen = np.zeros(np.shape(l2))
	cl2_tot = np.zeros(np.shape(l2))
	deltal = np.zeros(np.shape(l2))

	idxs1 = np.where( np.logical_and( lmin < l2, l2 < lmax) )
	idxs2 = np.where( l2 <= lmin )
	idxs3 = np.where( l2 >= lmax )

	lowl = np.floor(l2).astype(int)
	highl = np.ceil(l2).astype(int)
	deltal[idxs1] = l2[idxs1] - lowl[idxs1]
	deltal[idxs2] = lmin - l2[idxs2]
	deltal[idxs3] = l2[idxs3] - lmax

	lowl -= lmin
	highl -= lmin

	cl2_tot[idxs1] = cl_tot[lowl[idxs1]] + deltal[idxs1] * (cl_tot[highl[idxs1]] - cl_tot[lowl[idxs1]])
	cl2_unlen[idxs1] = cl_unlen[lowl[idxs1]] + deltal[idxs1] * (cl_unlen[highl[idxs1]] - cl_unlen[lowl[idxs1]])

	cl2_tot[idxs2] = cl_tot[0] + deltal[idxs2] * (cl_tot[0] - cl_tot[1])
	cl2_unlen[idxs2] = cl_unlen[0] + deltal[idxs2] * (cl_unlen[0] - cl_unlen[1])
	cl2_tot[idxs3] = cl_tot[lmax-lmin] + deltal[idxs3]*(cl_tot[lmax-lmin] - cl_tot[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2)
	cl2_unlen[idxs3] = cl_unlen[lmax-lmin] + deltal[idxs3]*(cl_unlen[lmax-lmin] - cl_unlen[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2)

	return cl2_unlen, cl2_tot

def noise_kernel(theta, l1, L, field, cl_unlen, cl_len, cl_tot, lmin, lmax):
    Ldotl1 = L * l1 * np.cos(theta)
    Ldotl2 = L**2 - Ldotl1
    l1dotl2= Ldotl1 - l1**2
    l2 = np.sqrt( L**2 + l1**2 - 2.*Ldotl1 )
    l2[ np.where(l2 < 0.000001) ] = 0.000001 ## Avoid nasty things
    cos_phi = l1dotl2 / ( l1 * l2 )
    phi = np.arccos( cos_phi )
    cos_2phi = np.cos( 2 * phi )
    sin_2phi = np.sin( 2 * phi )

    if field == 'tt':
        cl1_len   = cl_len['tt'][l1]
        cl1_tot   = cl_tot['tt'][l1]
        cl2_len, cl2_tot = get_cl2(cl_len['tt'], cl_tot['tt'], l2, lmin, lmax)
        kernel    = ( cl1_len * Ldotl1 + cl2_len * Ldotl2 )**2 / (2. * cl1_tot * cl2_tot)

    elif field == 'te':
        cl2TT = get_cl2(cl_len['tt'], cl_tot['tt'], l2, lmin, lmax)[1]
        cl2_unlen, cl2_tot = get_cl2(cl_len['te'], cl_tot['te'], l2, lmin, lmax)
        cl2EE = get_cl2(cl_len['ee'], cl_tot['ee'], l2, lmin, lmax)[1]
        cl1_unlen = cl_len['te'][l1]

        f_l1l2 = cl1_unlen * cos_2phi * Ldotl1 + cl2_unlen * Ldotl2
        f_l2l1 = cl2_unlen * cos_2phi * Ldotl2 + cl1_unlen * Ldotl1
        F_l1l2 = (cl_tot['ee'][l1] * cl2TT * f_l1l2 - cl_tot['te'][l1] * cl2_tot * f_l2l1)/(cl_tot['tt'][l1]*cl2EE*cl_tot['ee'][l1]*cl2TT - (cl_tot['te'][l1]*cl2_tot)**2)
        kernel = f_l1l2 * F_l1l2

    elif field == 'ee':
        cl1_tot = cl_tot['ee'][l1]
        cl1_unlen = cl_len['ee'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_len['ee'], cl_tot['ee'], l2, lmin, lmax)
        kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)

    elif field == 'eb':
        cl1_unlen = cl_len['ee'][l1]
        cl1_len = cl_len['ee'][l1]
        cl1EE = cl_tot['ee'][l1]
        cl2_unlen, cl2BB = get_cl2(cl_len['bb'], cl_tot['bb'], l2, lmin, lmax)
        f_l1l2 = (cl1_unlen * Ldotl1 - cl2_unlen * Ldotl2) * sin_2phi
        kernel = (f_l1l2)**2 / (cl1EE * cl2BB)

    elif field == 'tb':
        cl1TT = cl_tot['tt'][l1]
        cl2BB = get_cl2(cl_len['bb'], cl_tot['bb'], l2, lmin, lmax)[1]
        cl1_unlen = cl_len['te'][l1]
        kernel = (cl1_unlen * Ldotl1 * sin_2phi )**2 / (cl1TT * cl2BB)

    elif field == 'bb':
        cl1_tot = cl_tot['bb'][l1]
        cl1_unlen = cl_len['bb'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_len['bb'], cl_tot['bb'], l2, lmin, lmax)
        kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)
    kernel *= (l1 * (2. * np.pi)**(-2.))
    return kernel






def get_lensing_noise(ells, cl_len, cl_unlen, nl, fields,lmin,lmax):
    result ={}

    cl_tot = {}
    n_Ls   = 200
    LogLs  = np.linspace(np.log(1.),np.log(4000), n_Ls)
    Ls     = np.unique(np.floor(np.exp(LogLs)).astype(int))

    for field in fields:
        try:
            cl_tot[field] = cl_len[field]+nl[field]
        except:
            pass


    for field in fields:
        integral=[]
        for L in Ls:
            N = 100
            thetas = np.linspace(0.,2*np.pi,N)
            dtheta= 2.*np.pi/N
            Theta, Ells = np.meshgrid(thetas,np.arange(lmin,lmax))
            kernel_grid = noise_kernel(Theta, Ells, L, field, cl_unlen, cl_len, cl_tot, min(ells), max(ells))
            integral+=[dtheta * np.sum(np.sum(kernel_grid, axis = 0), axis = 0)]
        result[field] = 1./ np.asarray(integral)

    return Ls, result


## Read in ells, unlensed, lensed and noise spectra (TT, EE, ...)
## and define which spectra we have measured (fields)


print 'Evaluating reconstruction noise for fields %s, noise level %f muK/arcmin and %s arcmin sq beam'%(str(fields),noiseUkArcmin,thetaFWHMarcmin)

if l_max_T!=l_max_P:
    lmax='mixedlmax_%d_%d'%(l_max_T,l_max_P)
else:
    lmax=str(l_max_T)


try:
  parameter,cl_unl,cl_len=pickle.load(open(inputpath+'%s'%class_file,'r'))
except:
  closmo = Class()
  params['output']='tCl lCl pCl mPk'
  params['lensing']='yes'
  if nl:
    params['non linear']="halofit"
    tag+="_nl"
  else:
    params['non linear']=" "
  params['tol_perturb_integration']=1e-6
  params['perturb_sampling_stepsize']=0.01
  params['l_max_scalars']=7000
  closmo.set(params)


  print "Calculalating Cls... with settings", params

  closmo.compute()

  cl_len=closmo.lensed_cl(6999)

  cl_unl=closmo.raw_cl(6999)


  pickle.dump([params,cl_unl,cl_len],open(inputpath+'%s'%class_file,'w'))

cl, nl = {}, {}

ll=cl_unl['ell']
cl_phiphi=cl_len['pp'][ll]


thetaFWHM = thetaFWHMarcmin*np.pi/(180.*60.) #beam FWHM in rad
deltaT = noiseUkArcmin/thetaFWHMarcmin # noise variance per unit area
nlI_T = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2 #beam deconvolved noise relative to CMB temperature
nlI_pol = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2


nlI_T[0:l_min]=1e20
nlI_T[l_max_T::]=1e20

nlI_pol[0:l_min]=1e20
nlI_pol[l_max_P::]=1e20

#beam deconvolved noise
nl['tt']  = nlI_T
nl['te']  = np.zeros(len(nlI_T))
nl['tb']  = np.zeros(len(nlI_T))
nl['ee']  = 2*nlI_pol
nl['bb']  = 2*nlI_pol
nl['eb']  = np.zeros(len(nlI_T))


Ls, NL_KK = get_lensing_noise(ll, cl_len,cl_unl, nl, fields,L_min,L_max)


#optionally: divide by 2.5 to approximate noise on iterative estimators
if div:
    print 'Dividing EB by factor 2.5!'
    NL_KK['eb']*=1./2.5
    no_div='div25'
else:
    no_div='nodiv'

#minimum variance combination
MV_noise=0
for f in fields:
	if f!='bb':
			MV_noise+=1./NL_KK[f]
MV_noise=1./MV_noise
NL_KK['mv']=MV_noise

filename = out_path+'%s_N0_%s_%d_%d%d_%s%s.pkl'%(tag,lmax,l_min,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)
print out_path+'%s_N0_%s_%d_%d%d_%s%s.pkl'%(tag,lmax,l_min,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_)

pickle.dump([Ls,NL_KK],open(filename,'w'))

plotpath=None#path to save testplot


#bla=np.loadtxt(path+'noise_vanessa-_bw_10_dT_10.txt',delimiter=' ',comments='#' )
#bla=bla.T
colors=['b','r','c','g','y','m']
plt.figure(figsize=(9,7))
plt.loglog(ll,1./4.*(ll*(ll+1.))**2*cl_phiphi,color=colors[0], label=r'$C_L^{\kappa\kappa}$')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['tt'],color=colors[1],label='tt')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['ee'],color=colors[2],label='ee')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['te'],color=colors[3],label='te')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['eb'],color=colors[4],label='eb')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['tb'],color=colors[5],label='tb')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* MV_noise,color='black',label='MV')
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')

plt.xlim(2, 2000)
plt.ylim(1.e-9,5.e-6)
plt.grid()
plt.legend(loc='best',ncol=4,frameon=False, columnspacing=0.8)
plt.xlabel(r'$L$')
plt.ylabel(r'$N_L^{\kappa\kappa}$')
#plt.savefig(plotpath+'noise_kk_%s_%s_%d_%d%d_%s%s.pdf'%(tag,lmax,l_min,10*noiseUkArcmin,10*thetaFWHMarcmin,no_div,nl_))
