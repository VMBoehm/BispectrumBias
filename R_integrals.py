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
    sin_phi = np.sin(phi)
    cos_2phi = np.cos( 2 * phi )
    sin_2phi = np.sin( 2 * phi )
    
    kernel={}

    if field == 'tt': 
        cl1_len   = cl_len['tt'][l1]
        cl1_unlen = cl_unlen['tt'][l1]
        cl1_tot   = cl_tot['tt'][l1]
        cl2_len, cl2_tot = get_cl2(cl_len['tt'], cl_tot['tt'], l2, lmin, lmax)
        g_    = (cl1_len * Ldotl1 + cl2_len * Ldotl2) / (2. * cl1_tot * cl2_tot)
        kernel['perp']  = g_*(sin_phi*l1)**2*cl1_unlen
        kernel['para']  = g_*(cos_phi*l1)**2*cl1_unlen
        
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
        
    for tag in ['perp','para']:  
        kernel[tag]*= (l1 * (2. * np.pi)**(-2.))
    return kernel		

		

	


def get_lensing_noise(ells, cl_len, cl_unlen, nl, fields,lmin,l_max_T,lmax_P):
    result ={}
    
    cl_tot = {}
    n_Ls   = 200
    LogLs  = np.linspace(np.log(1.),np.log(max(ells)+0.1), n_Ls)
    Ls     = np.unique(np.floor(np.exp(LogLs)).astype(float))
    
    for field in fields:
        try:
            cl_tot[field] = cl_len[field]+nl[field]
        except:
            pass

    
    for field in fields:
        result[field]={}
        if field in ['ee','bb','eb']:
            lmax=l_max_P
        else:
            lmax=l_max_T
        integral={}
        integral['perp']=[]
        integral['para']=[]
        for L in Ls:
            N = 200
            thetas = np.linspace(0.,2*np.pi,N)
            dtheta= 2.*np.pi/N
            Theta, Ells = np.meshgrid(thetas,np.arange(lmin,lmax+1))
            kernel_grid = noise_kernel(Theta, Ells, L, field, cl_unlen, cl_len, cl_tot, lmin, lmax)
            integral['perp']+=[dtheta * np.sum(np.sum(kernel_grid['perp'], axis = 0), axis = 0)]
            integral['para']+=[dtheta * np.sum(np.sum(kernel_grid['para'], axis = 0), axis = 0)]
        
        for tag in ['perp','para']:
            result[field][tag]=np.asarray(integral[tag])
    return Ls, result


## Read in ells, unlensed, lensed and noise spectra (TT, EE, ...)
## and define which spectra we have measured (fields)

params=Cosmo.Planck2015_TTlowPlensing
tag=params[0]['name']
fields = ['tt']#,'te','ee','eb','bb','tb']

thetaFWHMarcmin = 1. #beam FWHM
noiseUkArcmin = 1.#eval(sys.argv[1]) #Noise level in uKarcmin
l_max_T       = 3000
l_max_P       = 5000
TCMB          = 2.7255e6

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
nlI[3000::]=1e10

#beam deconvolved noise
nl['tt']  = nlI
nl['te']  = np.zeros(len(nlI))
nl['tb']  = np.zeros(len(nlI))
nl['ee']  = 2*nlI
nl['bb']  = 2*nlI
nl['eb']  = np.zeros(len(nlI))

path='./R_files/'

filename = path+'R_'+str(int(noiseUkArcmin*10))+str(int(thetaFWHMarcmin*10))+'_%s_nl.pkl'%tag
try:
    Ls,results = pickle.load(open(filename,'r'))
    assert(False)
except:
    Ls,results = get_lensing_noise(ll, cl_len,cl_unl, nl, fields,2,l_max_T,l_max_P)

    pickle.dump([Ls,results],open(filename,'w'))
    
    
plt.figure()
plt.plot(Ls,Ls**(-2)*results['tt']['perp'])
plt.plot(Ls,Ls**(-2)*results['tt']['para'])
plt.xlim(100,3000)
plt.show()