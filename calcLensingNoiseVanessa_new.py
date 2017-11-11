## calcLensingNoise.py
## Based on 'LensingNoise.py' in OxFish_New
## Calculate the minimum variance noise spectrum 
## Computes A_L = N^0 and plots (ell+1)^2*N^0/(2 pi) and compares with C^{dd}
## uses lensed power spectra
## make sure you are using correct version of Cosmology/ copy recent version from downloaded
## choose settings: cosmology, lmax before running


from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as spline
import sys
import pickle

from classy import Class
import Cosmology as Cosmo

## Format labels to Latex font
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':16})


def get_del_lnCl_del_lnl(lvec, Cl, test_plots=False):
	
    # TODO: use h^4 central finite difference
    #Cl[np.where(Cl==0)]=1.#np.nan
    #lvec[np.where(lvec==0)]=1#np.nan					
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
	
	cl2_tot[idxs2] = 1e10#cl_tot[0] + deltal[idxs2] * (cl_tot[0] - cl_tot[1]) 
	cl2_unlen[idxs2] = 0.#cl_unlen[0] + deltal[idxs2] * (cl_unlen[0] - cl_unlen[1])
	cl2_tot[idxs3] = 1e10#cl_tot[lmax-lmin] + deltal[idxs3]*(cl_tot[lmax-lmin] - cl_tot[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2) 
	cl2_unlen[idxs3] = 0. #cl_unlen[lmax-lmin] + deltal[idxs3]*(cl_unlen[lmax-lmin] - cl_unlen[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2)
	
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
#        cl1_len   = cl_len['tt'][l1]
        cl1_unlen = cl_unlen['tt'][l1]
        cl1_tot   = cl_tot['tt'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_unlen['tt'], cl_tot['tt'], l2, lmin, lmax)
        kernel    = ( cl1_unlen * Ldotl1 + cl2_unlen * Ldotl2 )**2 / (2. * cl1_tot * cl2_tot)
    elif field == 'te': 
        cl2TT = get_cl2(cl_unlen['tt'], cl_tot['tt'], l2, lmin, lmax)[1]
        cl2_unlen, cl2_tot = get_cl2(cl_unlen['te'], cl_tot['te'], l2, lmin, lmax)
        cl2EE = get_cl2(cl_unlen['ee'], cl_tot['ee'], l2, lmin, lmax)[1]
        cl1_unlen = cl_unlen['te'][l1]
        f_l1l2 = cl1_unlen * cos_2phi * Ldotl1 + cl2_unlen * Ldotl2
        f_l2l1 = cl2_unlen * cos_2phi * Ldotl2 + cl1_unlen * Ldotl1
        F_l1l2 = (cl_tot['ee'][l1] * cl2TT * f_l1l2 - cl_tot['te'][l1] * cl2_tot * f_l2l1)/(cl_tot['tt'][l1]*cl2EE*cl_tot['ee'][l1]*cl2TT - (cl_tot['te'][l1]*cl2_tot)**2)
        kernel = f_l1l2 * F_l1l2   
        
    elif field == 'ee':
        cl1_tot = cl_tot['ee'][l1]
        cl1_unlen = cl_unlen['ee'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_unlen['ee'], cl_tot['ee'], l2, lmin, lmax)
        kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)
        
    elif field == 'eb':
        cl1_unlen = cl_unlen['ee'][l1]
        cl1_len = cl_len['ee'][l1]
        cl1EE = cl_tot['ee'][l1]
        cl2_unlen, cl2BB = get_cl2(cl_unlen['bb'], cl_tot['bb'], l2, lmin, lmax)
        f_l1l2 = (cl1_unlen * Ldotl1 - cl2_unlen * Ldotl2) * sin_2phi
        kernel = (f_l1l2)**2 / (cl1EE * cl2BB)

    elif field == 'tb': 
        cl1TT = cl_tot['tt'][l1]
        cl2BB = get_cl2(cl_unlen['bb'], cl_tot['bb'], l2, lmin, lmax)[1]
        cl1_unlen = cl_unlen['te'][l1]
        kernel = (cl1_unlen * Ldotl1 * sin_2phi )**2 / (cl1TT * cl2BB)

    elif field == 'bb':
        cl1_tot = cl_tot['bb'][l1]
        cl1_unlen = cl_unlen['bb'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_unlen['bb'], cl_tot['bb'], l2, lmin, lmax)
        kernel = ( (cl1_unlen * Ldotl1 + cl2_unlen*Ldotl2) * cos_2phi ) **2 / (2 * cl1_tot * cl2_tot)        
    kernel *= (l1 * (2. * np.pi)**(-2.))
    return kernel		

		

	


def get_lensing_noise(ells, cl_len, cl_unlen, nl, fields,lmin):
    result ={}
    
    cl_tot = {}
    n_Ls  = 200
    LogLs = np.linspace(np.log(1.),np.log(max(ells)+0.1), n_Ls)
    Ls = np.unique(np.floor(np.exp(LogLs)).astype(int))
    
    for field in fields:
        try:
            cl_tot[field] = cl_len[field]+nl[field]
        except:
            pass

    
    for field in fields:
        if field in ['ee','bb','eb']:
            lmax=5000
        else:
            lmax=3000
        integral=[]
        for L in Ls:
            N = 100
            thetas = np.linspace(0.,2*np.pi,N)
            dtheta= 2.*np.pi/N
            Theta, Ells = np.meshgrid(thetas,np.arange(lmin,lmax+1))
            kernel_grid = noise_kernel(Theta, Ells, L, field, cl_unlen, cl_len, cl_tot, lmin, lmax)
            integral+=[dtheta * np.sum(np.sum(kernel_grid, axis = 0), axis = 0)]
        result[field] = 1./ np.asarray(integral)
        
    return Ls, result


## Read in ells, unlensed, lensed and noise spectra (TT, EE, ...)
## and define which spectra we have measured (fields)

params=Cosmo.Planck2013_TempLensCombined
tag='Planck2013'#params[0]['name']
fields = ['tt','te','ee','eb','bb','tb']

thetaFWHMarcmin = 5. #beam FWHM
noiseUkArcmin = 30.#eval(sys.argv[1]) #Noise level in uKarcmin
TCMB = 2.726e6 #CMB temp in uK

print 'Evaluating reconstruction noise for fields %s, noise level %f muK/arcmin and %s arcmin sq beam'%(str(fields),noiseUkArcmin,thetaFWHMarcmin)

try:
    Parameter,cl_unl,cl_len=pickle.load(open('/home/traveller/Documents/Projekte/LensingBispectrum/class_outputs/class_cls_%s_nl.pkl'%tag,'r'))
except:
    l_max=5000
    print 'class_cls_%s_nl.pkl not found...'%tag
    cosmo = Cosmo.Cosmology(Params=params, Limber=False, lmax=l_max, mPk=False)
    closmo=Class()
    cosmo.class_params['output']= 'lCl tCl pCl'
    cosmo.class_params['non linear'] = "halofit"
    cosmo.class_params['lensing'] = 'yes'
    closmo.set(cosmo.class_params)
    print "Initializing CLASS..."
    closmo.compute()
    print "sigma8:", closmo.sigma8()

    cl_unl=closmo.raw_cl(l_max)
    cl_len=closmo.lensed_cl(l_max)
    pickle.dump([cosmo.class_params,cl_unl,cl_len],open('/home/traveller/Documents/Projekte/LensingBispectrum/class_outputs/class_cls_%s.pkl'%tag,'w'))
    print 'Done!'


cl, nl = {}, {}
#cl_len['tt']=cl_len['tt']
#cl_unl['tt']=cl_unl['tt']
#cl['ee']=cl_len['ee']
#cl['EE_unlen']=cl_unl['ee'][ells]
#cl['te']=cl_len['te'][ells]
#cl['TE_unlen']=cl_unl['te'][ells]
#cl['BB']=cl_len['bb'][ells]
ll=cl_unl['ell']
#assert(min(ll)==min(ells))




#cl['BB_unlen']=np.zeros(len(cl_len['bb']))
#cl['EB_unlen']=np.zeros(len(cl_len['bb']))
#cl['TB_unlen']=np.zeros(len(cl_len['bb']))
#cl['EB']=np.zeros(len(cl_len['bb']))
#cl['TB']=np.zeros(len(cl_len['bb']))
#
cl_phiphi=cl_len['pp'][ll]


thetaFWHM = thetaFWHMarcmin*np.pi/(180.*60.) #beam FWHM in rad
deltaT = noiseUkArcmin/thetaFWHMarcmin # noise variance per unit area
nlI = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2 #beam deconvolved noise relative to CMB temperature

nlI[0:2]=1e10
#nlI[3001:-1]=1e10
#beam deconvolved noise
nl['tt']  = nlI
nl['te']  = np.zeros(len(nlI))
nl['tb']  = np.zeros(len(nlI))
#nl['te'][0:2]=1e10

nl['ee']  = 2*nlI
nl['bb']  = 2*nlI
nl['eb']  = np.zeros(len(nlI))

#plt.figure()
#plt.loglog(ells*(ells+1.)*cl['tt']/2./np.pi)
#plt.loglog(ells*(ells+1.)*nlI/2./np.pi)
#plt.ylim(1e-15,1e-9)
#plt.xlim(2,3000)
#plt.show()

#thetaFWHM = thetaFWHMarcmin*np.pi/(180.*60.)
#deltaT    = np.sqrt(2)*noiseUkArcmin/thetaFWHMarcmin
#ll        = ells
#nlI       = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2
#
#nlI[0:1]    =10.**20
#nlI[3001:-1]=10.**20
#nl['ee'] = nlI
#nl['BB'] = nlI
#nl['EB'] = 0.
#nl['TB'] = 0.
#nl['te'] = 0.
#
#
#
#	## Define which scales to use in reconstruction.
#q = { 	'L_lens_min' : 2,
#			'L_lens_max_temp' : 2000,
#			'L_lens_max_pol' : 2000,}

Ls, NL_KK = get_lensing_noise(ll, cl_len,cl_unl, nl, fields,lmin=2)
#N0_lim    = get_lowL_limit(cl['TT_unlen'][2::],cl['tt'][2::], cl['tt'][2::]+nl['tt'][2::], ells[2::],Ls)


#print 'Dividing EB by factor 2.5!'
#NL_KK['EB']*=1./2.5
#MV_noise=0
#for f in fields:
#	if f!='BB':
#			MV_noise+=1./NL_KK[f]
#
#MV_noise=1./MV_noise
#pickle.dump([Ls,NL_KK,MV_noise],open('../results/lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl.pkl'%tag,'w'))
#	pickle.dump([Ls,N0_lim],open('lensNoisePowerlim'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_'+str(int(lcut))+'.pkl','w'))
#plt.loglog(ells,ells*(ells+1.)*cl_phiphi/2./np.pi, label=r'$C_\ell^{\phi\phi}$')
#plt.xlim(2, 3000)
#plt.ylim(2.e-17,2.e-7)
#plt.legend(loc='best')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$l^2 N_l^{\phi\phi}/2 \pi$')
#plt.savefig('noise_phiphi'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_'+str(int(lcut))+'.png')

bla=np.loadtxt('/home/traveller/Documents/Projekte/LensingBispectrum/CosmoCodes/N0files/noise_ext4_bw_50_dT_300.txt',delimiter=' ',comments='#' )
bla=bla.T
plt.figure(figsize=(9,7))
print Ls
plt.loglog(ll,1./4.*(ll*(ll+1.))**2*cl_phiphi, label=r'$C_L^{\kappa\kappa}$')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['tt'],label='tt')
plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* NL_KK['ee'],label='ee')
plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* NL_KK['te'],label='te')
plt.semilogy(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* NL_KK['eb'],label='eb')
plt.semilogy(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* NL_KK['tb'],label='tb')
plt.loglog(bla[0], bla[2],'--')
plt.loglog(bla[0], bla[3],'--')
plt.loglog(bla[0], bla[4],'--')
plt.loglog(bla[0], bla[5],'--')
plt.loglog(bla[0], bla[6],'--')
#plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* MV_noise,'k',lw=2,label='MV')
#plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* N0_lim ,label='lim')
plt.xlim(2, 2000)
plt.ylim(1.e-8,1.e-4)
plt.legend(loc='best',ncol=4,frameon=False, columnspacing=0.8)
plt.xlabel(r'$L$')
plt.ylabel(r'$N_L^{\kappa\kappa}$')
plt.savefig('noise_kk'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'match_chirag1_%s.png'%tag)


#plt.figure()
#plt.semilogx(Ls, (NL_KK['tt']-N0_lim)/NL_KK['tt'],label='tt')
##plt.loglog(Ls, ,label='TT lim')
#plt.xlim(2, 1000)
#plt.ylim(-2,2)
#plt.legend(loc='best',ncol=2)
plt.show()