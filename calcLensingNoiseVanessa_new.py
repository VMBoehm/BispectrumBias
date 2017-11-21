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
        cl1_unlen = cl_len['tt'][l1]
        cl1_tot   = cl_tot['tt'][l1]
        cl2_unlen, cl2_tot = get_cl2(cl_len['tt'], cl_tot['tt'], l2, lmin, lmax)
        kernel    = ( cl1_unlen * Ldotl1 + cl2_unlen * Ldotl2 )**2 / (2. * cl1_tot * cl2_tot)
        
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

		

	


def get_lensing_noise(ells, cl_len, cl_unlen, nl, fields,lmin,l_max_T,lmax_P):
    result ={}
    
    cl_tot = {}
    n_Ls   = 200
    LogLs  = np.linspace(np.log(1.),np.log(max(ells)+0.1), n_Ls)
    Ls     = np.unique(np.floor(np.exp(LogLs)).astype(int))
    
    for field in fields:
        try:
            cl_tot[field] = cl_len[field]+nl[field]
        except:
            pass

    
    for field in fields:
        if field in ['ee','bb','eb']:
            lmax=l_max_P
        else:
            lmax=l_max_T
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

params=Cosmo.Planck2015_TTlowPlensing
tag=params[0]['name']
fields = ['tt','te','ee','eb','bb','tb']

thetaFWHMarcmin = 1. #beam FWHM
noiseUkArcmin = 1.#eval(sys.argv[1]) #Noise level in uKarcmin
l_max_T       = 3000
l_max_P       = 5000
TCMB          = 2.7255e6
div           = False #divide EB by factor of 2.5

print 'Evaluating reconstruction noise for fields %s, noise level %f muK/arcmin and %s arcmin sq beam'%(str(fields),noiseUkArcmin,thetaFWHMarcmin)

try:
    Parameter,cl_unl,cl_len=pickle.load(open('/home/traveller/Documents/Projekte/LensingBispectrum/class_outputs/class_cls_%s_nl.pkl'%tag,'r'))
    print 'class_cls_%s_nl.pkl'%tag
except:
    print 'class_cls_%s_nl.pkl not found...'%tag
    l_max = max(l_max_T, l_max_P)+2000
    cosmo = Cosmo.Cosmology(Params=params, Limber=False, lmax=l_max, mPk=False)
    closmo=Class()
    cosmo.class_params['output']= 'lCl tCl pCl'
    cosmo.class_params['non linear'] = "halofit"
    cosmo.class_params['lensing'] = 'yes'
    closmo.set(cosmo.class_params)
    print "Initializing CLASS with", cosmo.class_params
    closmo.compute()
    print "sigma8:", closmo.sigma8()
    cl_unl=closmo.raw_cl(l_max)
    cl_len=closmo.lensed_cl(l_max)
    pickle.dump([cosmo.class_params,cl_unl,cl_len],open('/home/traveller/Documents/Projekte/LensingBispectrum/class_outputs/class_cls_%s_nl.pkl'%tag,'w'))
    print 'Dumped class cls under %s'%('/home/traveller/Documents/Projekte/LensingBispectrum/class_outputs/class_cls_%s_nl.pkl'%tag)


cl, nl = {}, {}

ll=cl_unl['ell']
cl_phiphi=cl_len['pp'][ll]


thetaFWHM = thetaFWHMarcmin*np.pi/(180.*60.) #beam FWHM in rad
deltaT = noiseUkArcmin/thetaFWHMarcmin # noise variance per unit area
nlI = (deltaT*thetaFWHM)**2*np.exp(ll*(ll+1.)*thetaFWHM**2/(8.*np.log(2.)))/TCMB**2 #beam deconvolved noise relative to CMB temperature

nlI[0:2]=1e10

#beam deconvolved noise
nl['tt']  = nlI
nl['te']  = np.zeros(len(nlI))
nl['tb']  = np.zeros(len(nlI))
nl['ee']  = 2*nlI
nl['bb']  = 2*nlI
nl['eb']  = np.zeros(len(nlI))

path='/home/traveller/Documents/Projekte/LensingBispectrum/CosmoCodes/N0files/'

print path+'lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl.pkl'%tag
try:
    Ls,NL_KK,MV_noise=pickle.load(open(path+'lensNoisePower1010_Planck2013_nl.pkl','r'))
    
except:
    print 'lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl.pkl'%tag, 'not found'
    Ls, NL_KK = get_lensing_noise(ll, cl_len,cl_unl, nl, fields,2,l_max_T,l_max_P)
    
#N0_lim    = get_lowL_limit(cl['TT_unlen'][2::],cl['tt'][2::], cl['tt'][2::]+nl['tt'][2::], ells[2::],Ls)


if div:
    print 'Dividing EB by factor 2.5!'
    NL_KK['EB']*=1./2.5
    
MV_noise=0
for f in fields:
	if f!='BB':
			MV_noise+=1./NL_KK[f]
MV_noise=1./MV_noise


pickle.dump([Ls,NL_KK,MV_noise],open(path+'lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl_lensedCls.pkl'%tag,'w'))
#	pickle.dump([Ls,N0_lim],open('lensNoisePowerlim'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_'+str(int(lcut))+'.pkl','w'))
#plt.loglog(ells,ells*(ells+1.)*cl_phiphi/2./np.pi, label=r'$C_\ell^{\phi\phi}$')
#plt.xlim(2, 3000)
#plt.ylim(2.e-17,2.e-7)
#plt.legend(loc='best')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$l^2 N_l^{\phi\phi}/2 \pi$')
#plt.savefig('noise_phiphi'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_'+str(int(lcut))+'.png')
tag='Planck2013'
Ls2,NL_KK2,MV_noise2=pickle.load(open(path+'lensNoisePower'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl_lensedCls.pkl'%tag,'r'))


colors=['b','r','c','g','y','m']
bla=np.loadtxt(path+'noise_vanessa-_bw_10_dT_10.txt',delimiter=' ',comments='#' )
bla=bla.T
plt.figure(figsize=(9,7))
print Ls
plt.loglog(ll,1./4.*(ll*(ll+1.))**2*cl_phiphi,color=colors[0], label=r'$C_L^{\kappa\kappa}$')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK2['tt'],color=colors[1],label='tt')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK2['ee'],color=colors[2],label='ee')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK2['te'],color=colors[3],label='te')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK2['eb'],color=colors[4],label='eb')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK2['tb'],color=colors[5],label='tb')
plt.loglog(bla[0], bla[2],'k--')
plt.loglog(bla[0], bla[3],'k--')
plt.loglog(bla[0], bla[4],'k--')
plt.loglog(bla[0], bla[5],'k--')
plt.loglog(bla[0], bla[6],'k--')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['tt'],'k--',label='cl unlen')#,label='tt')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['ee'],'k:')#,label='ee')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['te'],'k:')#,label='te')
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['eb'],'k:')#,label='eb') 
plt.loglog(Ls, 1./4.*(Ls*(Ls + 1.))**2.* NL_KK['tb'],'k:')#,label='tb')
t = lambda l: 0.25*(l*(l+1.))**2
data1= pickle.load(open(path+'noise_levels_n10_beam10_lmax3000.pkl','r'))
data2= pickle.load(open(path+'noise_levels_n10_beam10_lmax5000.pkl','r'))
ls = data1[0]
plt.loglog(ls, t(ls)*data1[1],ls='-.',lw=2,color=colors[1],label='TT')
plt.loglog(ls, t(ls)*data1[3],ls='-.',lw=2,color=colors[3],label='TE')
plt.loglog(ls, t(ls)*data1[4],ls='-.',lw=2,color=colors[5],label='TB')
ls = data2[0]
plt.loglog(ls, t(ls)*data2[2],ls='-.',lw=2,color=colors[2],label='EE')
plt.loglog(ls, t(ls)*data2[5],ls='-.',lw=2,color=colors[4],label='EB')
plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* MV_noise,'k-',lw=2,label='MV unlen')
plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* MV_noise2,'k--',lw=2,label='MV len')
#plt.loglog(Ls, 1./4.*(Ls + 1.)**2.*Ls**2.* N0_lim ,label='lim')
plt.xlim(2, 2000)
plt.ylim(3.e-9,5.e-6)
plt.legend(loc='best',ncol=4,frameon=False, columnspacing=0.8)
plt.xlabel(r'$L$')
plt.ylabel(r'$N_L^{\kappa\kappa}$')
plt.savefig('noise_kk'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'match_quicklens_%s.png'%tag)


N02015={}
data1a=pickle.load(open(path+'noise_levels_n7_beam30_lmax2999_Namikawa.pkl','r'))
data2a=pickle.load(open(path+'noise_levels_n7_beam30_lmax4999_Namikawa.pkl','r'))
N02015['ls']=data1a[0]
print min(data1a[0])
N02015['tt']=data1a[1]
N02015['te']=data1a[3]
N02015['tb']=data1a[4]
N02015['ee']=np.interp(data1a[0],data2a[0],np.real(data2a[2]))
N02015['eb']=np.interp(data1a[0],data2a[0],np.real(data2a[5]))#/2.5))
#
N02015['MV']=np.zeros(len(N02015['eb']))
for ff in ['tt','tb','te','ee','eb']:
    N02015['MV']+=1./N02015[ff]
N02015['MV']=1./N02015['MV']
#
pickle.dump(N02015,open(path+'Namikawa_N0_fac25_mixedlmax_730.pkl','w'))
#
plt.figure(figsize=(8,6))
ls = data1a[0]
plt.loglog(ll,1./4.*(ll*(ll+1.))**2*cl_phiphi,color=colors[0], label=r'$C_L^{\kappa\kappa}$')
plt.loglog(ls, t(ls)*N02015['MV'],ls='-',lw=2,color='black',label='MV')
plt.loglog(ls, t(ls)*N02015['tt'],ls='-',lw=2,color=colors[1],label='TT')
plt.loglog(ls, t(ls)*N02015['te'],ls='-',lw=2,color=colors[3],label='TE')
plt.loglog(ls, t(ls)*N02015['tb'],ls='-',lw=2,color=colors[4],label='TB')
plt.loglog(ls, t(ls)*N02015['ee'],ls='-',lw=2,color=colors[2],label='EE')
plt.loglog(ls, t(ls)*N02015['eb'],ls='-',lw=2,color=colors[5],label='EB')
plt.xlim(2,2000)
plt.ylim(1e-9,1e-5)
plt.legend(loc='best',ncol=3)
plt.savefig('N0_PlanckTemLens2015_730_lmax3000-5000_div25_ql.png')
plt.show()