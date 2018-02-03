
from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
from scipy.interpolate import splev, splrep
import warnings
warnings.filterwarnings('error')
from Tools import simps
#from libc.stdlib cimport malloc, free
DTYPEF = np.float
ctypedef np.float_t DTYPEF_t

DTYPEI = np.int
ctypedef np.int_t DTYPEI_t



@cython.profile(False)
def get_cl2(cl_unlen, cl_tot, l2, lmin, lmax):
	"""
	Cl_TT interpolation taken from Blakes code
	lmin/lmax indicate range in which cls are given
	"""
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


@cython.profile(False)
def get_gl1L(cl_unlen,cl_len,cl_tot,mu1s, L, l1s,lmin,lmax):

	Ldotl1 = L * l1s * np.cos(mu1s)
	Ldotl2 = L**2 - Ldotl1
	l1dotl2 = Ldotl1 - l1s**2
	l2 = np.sqrt( L**2 + l1s**2 - 2*Ldotl1 )
	l2[ np.where(l2 < 0.000001) ] = 0.000001 ## Avoid nasty things

	assert(lmin==0)


	l1s=l1s.astype(int)
	cl1_unlen = cl_unlen[l1s]
	cl1_len = cl_len[l1s]
	cl1_tot = cl_tot[l1s]
	cl2_len, cl2_tot = get_cl2(cl_len, cl_tot, l2, lmin, lmax)
	weight =(cl1_len * Ldotl1 + cl2_len * Ldotl2 )/ (2. * cl1_tot * cl2_tot)

	return weight

@cython.boundscheck(False)
def get_bias(double L, double[:] l3s, unsigned int len_l, cl_fid, field , nl, int lmin, int lmax, double min_mu, double max_mu, unsigned int num,unsigned int rank,unsigned int size,splines,unsigned int N,unsigned int M,unsigned int K,unsigned int P):
	"""
	* L: double, value of L
	* l3s: arrray of double, values of l in interpolated bispec
	* len_l: len(l3s)
	* cl_fid: list of fiducial cls from class
	* field: string, for now 'TT'
	* nl: list of deconvolved noise power spectra
	* lmin/lmax: range for which cl_fids are given (for calulation of weights)
	* lmin/lmax: range for which cl_fids are given
	* num: L index
	* rank: rank of process
	* size: number of processes
	* splines: bispectrum interpolation splines
	* N,M,K,P: len(mu1s), len(mus), len(ls), len(l1s) for integration
	"""

	#indices for l1,mu1,l,l3ple1d
	cdef unsigned int i,j,k,m

	#start integration at
	minl = 2
	minl1= 2

	delta = int(P/size)

	if delta == 0:
		raise ValueError, 'to many processors for this length of l1! len(l1): %d size: %d'%(P,size)

	#get part of l1 for this process
	l1Min = minl1+rank*delta
	l1Max = minl1+(rank+1)*delta

	print "rank ", rank
	print "delta ", delta
	print "l-range: ", minl, K+minl
	print "l1-range: ", minl1, P+minl1
	#length of loop for this process
	cdef unsigned int pP = delta

	cdef np.ndarray[DTYPEF_t, ndim=1] l1s  = np.arange(l1Min,l1Max,dtype=DTYPEF)
	assert(len(l1s)==pP)
	cdef np.ndarray[DTYPEF_t, ndim=1] ls   = np.arange(minl,K+minl, dtype=DTYPEF)
	cdef np.ndarray[DTYPEF_t, ndim=1] mu1s = np.linspace(min_mu,max_mu,N,dtype=DTYPEF)
	cdef np.ndarray[DTYPEF_t, ndim=1] mus  = np.linspace(min_mu,max_mu,M,dtype=DTYPEF)

	#only needed if using summation instead of simpsons rule from scipy (tested, gives the same results)
	#cdef double diff = np.diff(mu1s)[0]
	#assert(np.diff(np.diff(mu1s)).all()<1e-15)

	#integrands, where possible in memoryview (for faster access)
	cdef np.ndarray[DTYPEF_t, ndim=1] integrand_mu = np.zeros(M,dtype=DTYPEF)
	cdef double[:] integrand_mu1                   = np.zeros(N,dtype=DTYPEF)
	cdef double[:] integrand_l                     = np.zeros(K,dtype=DTYPEF)
	cdef double[:] integrand_l1                    = np.zeros(pP,dtype=DTYPEF)

	#precompute cosines
	cdef np.ndarray[DTYPEF_t, ndim=1] cos_mu       = np.cos(mus)
	cdef np.ndarray[DTYPEF_t, ndim=1] sin_mu       = np.sin(mus)
	cdef double[:] cos_mu1                         = np.cos(mu1s)
	cdef double[:] sin_mu1                         = np.sin(mu1s)

	# get grids for mu1 x mu1
	mu1_, mu_ 		= np.meshgrid(mu1s, mus)
	diffmu 			= np.transpose(mu1_-mu_) #rows of constant mu1, i.e. diffmu[i]=mu1[i]-mu

	# get grids for ls x sin(mu)
	sin_mu_g, ls_g 	= np.meshgrid(sin_mu, ls) #shape: (len(ls),len(sin_mu))

	# get grids for ls x cos(mu)
	cos_mu_g, ls_g  = np.meshgrid(cos_mu, ls) #shape: (len(ls),len(sin_mu))

	assert(len(sin_mu_g[0])==M)
	assert(len(cos_mu_g[0])==M)
	assert(len(ls_g[0])==M)

	cdef np.ndarray[DTYPEF_t, ndim=1] sinmu_gf = sin_mu_g.flatten()

	cdef np.ndarray[DTYPEF_t, ndim=1] cosmu_gf = cos_mu_g.flatten()

	cdef np.ndarray[DTYPEF_t, ndim=1] ls_g2f   = (ls_g**2).flatten()

	cdef np.ndarray[DTYPEF_t, ndim=1] ls_gf    = ls_g.flatten()

	cdef np.ndarray[DTYPEF_t, ndim=1] cos_diff = np.zeros(M,dtype=DTYPEF) #cos diff for fixed mu1

	cdef np.ndarray[DTYPEF_t, ndim=2] cosdiff  = np.cos(diffmu)


	#tested this for M!=N
	assert(len(diffmu[0])==M)
	assert(len(cosdiff[0])==M)

	#initialize cl arrays
	# cl is a function of mu and l that is flattened for faster access
	cdef np.ndarray[DTYPEF_t, ndim=1] cl   = np.zeros(M*K,dtype=DTYPEF)
	cdef double[:] l_                      = np.zeros(M*K,dtype=DTYPEF)
	# cl_ is cl for fixed ls
	cdef np.ndarray[DTYPEF_t, ndim=1] cl_  = np.zeros(M,dtype=DTYPEF)
	cdef np.ndarray[DTYPEF_t, ndim=1] d    = np.zeros(len(ls_g),dtype=DTYPEF)
	# interpolate cl_tt, tested, use k=1
	cl_tot = np.asarray(cl_fid[field] + nl[field])
	cl_unlen = np.asarray(cl_fid[field+'_unlen'])
	cl_len   = np.asarray(cl_fid[field])
	if rank==0:
		print "range of interpolation for cls", 2, lmax
	ells_=np.arange(1,lmax+1)
	cl_unl_interp=splrep(ells_,cl_unlen[1::],k=1)
	# get bispectrum spline for fixed L
	#spline=splines[num*lenL]

	#get reconstruction weight (code snippets from N0 code)
	mu1grid, l1grid 	= np.meshgrid(mu1s, l1s)
	gl1L=get_gl1L(cl_unlen, cl_len, cl_tot,mu1grid,L,l1grid,lmin,lmax)
	assert(gl1L.shape==(pP,N))

	cdef double[:,:] weight = gl1L

	### interpolate bispectrum for ls and mus of grid
	bispec = np.zeros((len_l,M),dtype=DTYPEF)

	#interpolate in mus
	for k in range(len_l):
		bispec[k]=splev(mus,splines[k],ext=2)#value error if outside bounds
	bispec=np.transpose(bispec) #bispec[i] is now function of l for fixed mu

	bispec_ = np.zeros((M,K),dtype=DTYPEF)

	#interpolate in ls for fixed mu
	for m in range(M):
		bispec_i=splrep(l3s,bispec[m],k=1)
		bispec_[m]=splev(np.asarray(ls),bispec_i,ext=2)#value error if outside bounds

	#make bispec[i] function of mu again
	cdef np.ndarray[DTYPEF_t, ndim=2] bispec_new = np.transpose(bispec_)
	del bispec_

	#needed later in the loops for accesing correct entries in cl_
	cdef np.ndarray[DTYPEI_t, ndim=1] index = np.arange(M,dtype=int)

	cdef double l1,l,l2,l12,l1l,w,l1cos,l1sin,term1,cl1

	#bispectrum for fixed L, l
	cdef np.ndarray[DTYPEF_t, ndim=1] b = np.zeros(M,dtype=DTYPEF)

	#l1 loop
	for i in xrange(pP):
		l1=l1s[i]
		l12=l1*l1
		cl1=cl_unlen[int(l1)]
		#mu1 loop
		for j in xrange(N):
			l1cos=l1*cos_mu1[j]
			l1sin=l1*sin_mu1[j]
			term1=L*l1cos
			w=weight[i,j]
			cos_diff=cosdiff[j] #cosdiff for mu1 fixed
			#l_ is a function of l and mu
			#sqrt(l1^2+l^2-2*l*(l1sinmu1*sinmu+l1cosmu1*cosmu))=|l-l1|
			d=l12+ls_g2f-2.*ls_gf*(l1sin*sinmu_gf+l1cos*cosmu_gf)
			d[np.where((d<0.)&(abs(d)<1e-7))]=0.
			try:
				l_=np.sqrt(d)
			except:
				print "except!"
				print d[np.where((d<0.)&(abs(d)>=1e-7))]
				print sinmu_gf[np.where((d<0.)&(abs(d)>=1e-7))], cosmu_gf[np.where((d<0.)&(abs(d)>=1e-7))]
				print cos_mu1[j], sin_mu1[j], l1, ls_gf[np.where((d<0.)&(abs(d)>=1e-7))]
				l_=np.zeros(len(ls_g2f))
				raise Exception('expression in sqrt too negative')
			#CTT(|l1-l|)
			#print np.asarray(l_)
			cl=splev(l_,cl_unl_interp,ext=1)#return zero if outside bounds
			#print cl
			#l loop
			for k in xrange(K):
				l=ls[k]
				l2=l*l
				l1l=l1*l
				#bispectrum for given l
				b=bispec_new[k]
				cl_=cl[k*M+index] #picks out entries as function of mu
				#assert(np.allclose(cosmu_gf[k*M+index],cos_mu))
				#gl1L*CTT|l1-l|*(l1l cos(l1,l)-l^2)*(l1l*cos(l1,l)-l^2-Ll1 cosmu1+Ll cosmu)*B(L,l,mu)
				integrand_mu=l1l*w*b*(cl1*l1l*cos_diff*(term1-l1l*cos_diff)\
                                  -cl_*(l1l*cos_diff-l2)*(l2+term1-l1l*cos_diff-L*l*cos_mu)\
                                  )
				integrand_l[k]=simps(integrand_mu,mus)#factor of two because we are only integrating over half of the angle
			integrand_mu1[j]=simps(integrand_l,ls)
		integrand_l1[i]=simps(integrand_mu1,mu1s)

	return num, L, l1s ,np.asarray(integrand_l1)*((2.*np.pi)**(-4)), M, P


