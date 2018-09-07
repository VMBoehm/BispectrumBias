import numpy, math
import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate

import sys, os, argparse
pwd  = os.getcwd()
sys.path.append(pwd)
import cosmology, tools
import pk_pregen as PK

#If file path is not specified in command line arguments, use these
class Files:
    noisefile = "../data/RunPB/noise/noise_ext_bw_%02d_dT_%02d.txt"
    pkfile = "../data/RunPB/cambPB_pk.txt"
    classfile = None
    errorfile = None
    outpath = "../data/RunPB/"


class CosmoPar:
    M = 0.292
    L = 1-M
    H0 = 100.
    h = 0.69



class LensingZ:
    '''Do lensing cross spectra for a delta function source'''

    def __init__(self, z, dndz, cosmo = None, l = None, dz = 0.5, nz = 100, ggnorm = None, purelimber=False):

        #Parameters which are not set
        if cosmo is None:
            cosmo = cosmology.Cosmology(M = CosmoPar().M, pfile = Files().pkfile)
        if l is None:
            l = numpy.linspace(100, 3000, 2901)
        if dndz is None:
            dndz = lambda x: 1
        self.ggnorm = ggnorm
    
        self.z = z
        self.dndz = dndz
        self.cosmo = cosmo
        self.l = l
        self.dz = dz
        self.nz = nz
        self.purelimber=purelimber
        self._clsetup()

    def _clsetup(self):
        
        zmin = self.z - self.dz/2.
        zmax = self.z + self.dz/2.
        self.zs = numpy.linspace(zmin, zmax, self.nz)
        self.xis = self.cosmo.xia(z = self.zs)

        #Kernels
        self.kerg = self.kernel_g(z =  self.zs, dndz= self.dndz)
        self.norm = np.trapz(self.kerg, self.xis)
        if self.ggnorm is not None:
            #Useful for photometric when vary dndz but keep total N constant
            print('External gg-norm given = %0.3f, gg-norm estimated from dndz = %0.3f\
            Overriding estimated norm. Be vary!!!'%(self.ggnorm, self.norm))
            self.norm = self.ggnorm
        self.kerg /= self.norm

        self.kercmb = self.kernel_cmb(z =  self.zs)
        self.ggwts = self.kerg**2./self.xis**2.
        self.kgwts = self.kercmb*self.kerg/self.xis**2.
        self.dggwtsdN = 2*self.kerg/self.xis**2. * self.cosmo.Ha(z = self.zs)/self.norm
        self.dkgwtsdN = self.kercmb/self.xis**2. * self.cosmo.Ha(z = self.zs)/self.norm

        #Integration Mesh
        self.kmesh = numpy.zeros([self.l.size, self.zs.size])
        for foo in range(self.zs.size):
            if self.purelimber:
                self.kmesh[:, foo] = self.cosmo.k_xil(self.l, z = self.zs[foo])            
            else:
                self.kmesh[:, foo] = self.cosmo.k_xil(self.l + 0.5, z = self.zs[foo])            



    def kernel_cmb(self, z = None, a = None, zcmb=1100):
        '''Calculates the kernel for cmb
        return 3H0^2*M/2/a/c^2*(xi-xi_s)*xi/xi_s
        '''
        z, a = self.cosmo._za(z, a)
        xistar = self.cosmo.xia(z=zcmb) #redshift of cmb
        xi = self.cosmo.xia(z)
        f = xi*(xistar - xi)/(xistar)
        fac = 3*self.cosmo.H0**2*self.cosmo.M/(2*a) 
        
        return f*fac* self.cosmo.cin**2.#cin is 1/c due to H0 (km/s)



    def kernel_cmb_noxi(self, z = None, a = None, zcmb=1100):
        '''Calculates the kernel for cmb
        return 3H0^2*M/2/a/c^2*(xi-xi_s)/xi_s
        '''
        z, a = self.cosmo._za(z, a)
        xistar = self.cosmo.xia(z=zcmb) #redshift of cmb
        xi = self.cosmo.xia(z)
        f = (xistar - xi)/(xistar) #Not multiply by xi and so do not divide by xi^2 for weights
        fac = 3*self.cosmo.H0**2*self.cosmo.M/(2*a) 
        
        return f*fac* self.cosmo.cin**2.#cin is 1/c due to H0 (km/s)


    def kernel_g(self, z, dndz):
        '''
        '''
        h = self.cosmo.Ha(z = z)
        return  dndz(z) *h 


    def clzeff(self, kp, auto = False):
        '''Assume constant power spectrum across redshift bin'''
        #pval = interpolate(*kp)(self.kmesh)
        pval = numpy.interp(self.kmesh, *kp)
        if auto:
            cl = pval * self.ggwts
        else:
            cl = pval * self.kgwts

        return numpy.trapz(cl, self.xis, axis = -1)

    def dclzeffdN(self, kp, auto = False, mask = None):
        '''Derivative with respect to dNdz, constant PS at zeff'''
        #pval = interpolate(*kp)(self.kmesh)
        pval = numpy.interp(self.kmesh, *kp)
        if auto:
            cl = pval * self.dggwtsdN
        else:
            cl = pval * self.dkgwtsdN
        if mask is not None:
            cl *= mask

        return numpy.trapz(cl, self.xis, axis = -1)

    def clzeff2(self, kp, dndz2):
        '''Cross spectra with a different tracer
        Assume constant power spectrum across redshift bin'''
        #pval = interpolate(*kp)(self.kmesh)
        pval = numpy.interp(self.kmesh, *kp)
        #Kernels
        kerg2 = self.kernel_g(z =  self.zs, dndz= dndz2)
        norm2 = np.trapz(kerg2, self.xis)
        kerg2 /= norm2
        #print(kerg2, norm2)
        ggwts2 = self.kerg*kerg2/self.xis**2.
        cl = pval * ggwts2
        return numpy.trapz(cl, self.xis, axis = -1)

    def dclzeffdN2(self, kp, dndz2, mask = None):
        '''Derivative of cross spectra with different tracer
        Derivative with respect to first dndz
        Assume constant power spectrum across redshift bin'''
        #pval = interpolate(*kp)(self.kmesh)
        #pval = interpolate(*kp)(self.kmesh)
        pval = numpy.interp(self.kmesh, *kp)
        #Kernels
        kerg2 = self.kernel_g(z =  self.zs, dndz= dndz2)
        norm2 = np.trapz(kerg2, self.xis)
        kerg2 /= norm2
        dggwts2 =  kerg2 * self.cosmo.Ha(z = self.zs)/self.norm /self.xis**2. 
        cl = pval * dggwts2
        if mask is not None:
            cl *= mask
        return numpy.trapz(cl, self.xis, axis = -1)



    def clz(self, ikpz, auto = False):
        '''Interpolating function for P(z) across redshift bin'''
        pval = numpy.zeros_like(self.kmesh)
        for foo in range(self.nz):
            kp = ikpz(z = self.zs[foo])
            pval[:, foo] = numpy.interp(self.kmesh[:, foo], *kp)

        if auto:
            cl = pval * self.ggwts
        else:
            cl = pval * self.kgwts

        return numpy.trapz(cl, self.xis, axis = -1)


        
    def clkg31d_internal(self, z, ipkmz):
        zz = np.linspace(0, z, 100)
        kercmb = self.kernel_cmb_noxi(z = zz, zcmb = z)
        xis = self.cosmo.xia(z = zz)
        #kerwt = kercmb / xis**2 
        #since xis goes to 0, cancel this xis with xis in integral
        #and use kercmb instead of kercmbwt
        pval = np.zeros([self.l.size, zz.size])

        for foo in range(zz.size):
            if self.purelimber:
                kk = self.cosmo.k_xil(self.l, z = zz[foo])            
            else:
                kk = self.cosmo.k_xil(self.l + 0.5, z = zz[foo])            
            kp = ipkmz(z = zz[foo])
            pval[:, foo] = numpy.interp(kk, *kp)

        #return np.trapz(pval*kerwt**2*xis**2, xis, axis=-1)
        return np.trapz(pval*kercmb**2, xis, axis=-1)

    def clkg31d(self, ipkhz, ipkmz):
            
        intfac = np.array([self.clkg31d_internal(z, ipkmz) for z in self.zs])
        intfac = np.trapz(intfac/self.l, self.l, axis=-1)

        pval = numpy.zeros_like(self.kmesh)
        for foo in range(self.nz):
            kp = ipkhz(z = self.zs[foo])
            pval[:, foo] = numpy.interp(self.kmesh[:, foo], *kp)

        integrand = pval*self.kgwts*intfac
        return np.trapz(integrand, self.xis, axis=-1) *self.l**2/2/np.pi
        #return integrand, intfac

            
        



def example_runpb(z, dndz,lo):
    '''Do for RunPB spectra'''
    
    iz = z*100
    if lo:
        hh = np.loadtxt('../data/RunPB/hh_fine_z%03d_lo.txt'%iz, unpack = True)
        hm = np.loadtxt('../data/RunPB/hm_fine_z%03d_lo.txt'%iz, unpack = True)
    else:
        hh = np.loadtxt('../data/RunPB/hh_fine_z%03d.txt'%iz, unpack = True)
        hm = np.loadtxt('../data/RunPB/hm_fine_z%03d.txt'%iz, unpack = True)

    lenz = LensingZ(z = z, dndz = dndz)

    l = lenz.l
    clggzeff = lenz.clzeff(hh, auto = True)
    clkgzeff = lenz.clzeff(hm, auto = False)

    toret = [l, clggzeff, clkgzeff]
    return toret

    #toret = np.array([l, clggz, clggzeff, clkgz, clkgzeff]).T
    #return toret
    #numpy.savetxt('clrunpb_z%03d'%(z*100), toret, \
    #                  header =  "[l, clggz, clggzeff, clkgz, clkgzeff]")


def example_halofit(z, dndz):
    '''Do for Halofit spectra with growth bias'''
    
    klin, plin = np.loadtxt("../data/RunPB/cambPB_pk.txt", unpack= True)
    cosmo = args.cosmo
    bz = tools.BiasZ(cosmo).linear(b = 2, z0 = 1)

    hmz =   lambda z: [klin,  cosmo.pkanlin(z = z)[1]*bz(z)]
    hhz =   lambda z: [klin,  cosmo.pkanlin(z = z)[1]*bz(z)**2. ]
    hh = hhz(z)
    hm = hmz(z)

    lenz = LensingZ(z = z, dndz = dndz)

    l = lenz.l
    clggz = lenz.clz(hhz, auto = True)
    clkgz = lenz.clz(hmz, auto = False)
    clggzeff = lenz.clzeff(hh, auto = True)
    clkgzeff = lenz.clzeff(hm, auto = False)

    toret = [l, clggz, clggzeff, clkgz, clkgzeff]
    return toret

    #toret = np.array([l, clggz, clggzeff, clkgz, clkgzeff]).T
    #return toret
    #numpy.savetxt('clrunpb_z%03d'%(z*100), toret, \
    #                  header =  "[l, clggz, clggzeff, clkgz, clkgzeff]")


def example_lpt(z, dndz):
    '''Do for Halofit spectra with growth bias'''
    
    lpt = PK.Pk8CLEFT(z = z)
    fits = list(np.loadtxt('joint_fit_results.log')[2*z - 2, 1:-1])
    print(fits)
    xfits = fits[:5]
    afits = fits[:3] + fits[-2:]

    hh = lpt(afits, auto = True)
    hm = lpt(xfits, auto = False)
    
    lenz = LensingZ(z = z, dndz = dndz)

    l = lenz.l

    clggzeff = lenz.clzeff(hh, auto = True)
    clkgzeff = lenz.clzeff(hm, auto = False)

    toret = [l,  clggzeff,  clkgzeff]
    return toret

    #toret = np.array([l, clggz, clggzeff, clkgz, clkgzeff]).T
    #return toret
    #numpy.savetxt('clrunpb_z%03d'%(z*100), toret, \
    #                  header =  "[l, clggz, clggzeff, clkgz, clkgzeff]")



if __name__=="__main__":
    
    lo = False

    args = tools.parse_arguments(Files = Files)
    cosmo = args.cosmo
    dndz = tools.DnDz().func(args.survey)
    zg = args.zg
    dz = args.dz
    print('dz = %.2f'%args.dz)

    args.noisefile = "../data/RunPB/noise/noise_ext4_bw_%d_dT_%d.txt"%(args.beam*10, args.temp*10)
    print('Noisefile used = ',args.noisefile)
    noise = np.loadtxt(args.noisefile).T
    ell = noise[0]
    clkk = noise[1]
    
    #Noises
    nobj = tools.nbar(zg - dz/2., zg + dz/2., survey = args.survey)
    arcm = numpy.pi/60./180.
    shotnoise = 1/(nobj /arcm**2.)

    nkk = (1/noise[3] + 1/noise[4] + 1/noise[5] + 1/noise[6])**-1 
    print(np.isnan(nkk).sum())
    mask = np.isnan(nkk)
    nkk[mask] = interpolate(noise[0][~mask], nkk[~mask])(noise[0][mask])
    print(np.isnan(nkk).sum())

    #cls
    #RunPB
#    l, clggzeff, clkgzeff = example_runpb(z = zg, dndz = dndz, lo =lo)
#    clggz, clkgz = l*0 , l*0
#    if lo:
#        suffix = 'runpb-lo'
#        b1 = np.loadtxt('log_jointfit/joint_fit_results_lo.log')[2*zg-2, 1] + 1
#        args.b1 = b1
#    else:
#        suffix = 'runpb2'
#        b1 = np.loadtxt('log_jointfit/joint_fit_results.log')[2*zg-2, 1] + 1
#        args.b1 = b1
#    print('b1 for %s = '%(suffix), b1)
#
    #Halofit
    l, clggz, clggzeff, clkgz, clkgzeff = example_halofit(z = zg, dndz = dndz)
    suffix = 'halofit'
    
    #LPT
    #l, clggzeff,clkgzeff = example_lpt(z = zg, dndz = dndz)
    #shotnoise = 0
    #clggz, clkgz = l*0 , l*0
    #suffix = 'lpt'

    #mix
    #p1, p2 = np.where(ell == l[0])[0][0], np.where(ell == l[-1])[0][0]
    #clkk[p1:p2+1]
    #nkk = nkk[p1:p2+1]
    clkk = interpolate(ell, clkk)(l)
    nkk = interpolate(ell, nkk)(l)
    #clkk[p1:p2+1]
    #nkk = nkk[p1:p2+1]
    
    varkgz = ((clkk + nkk)*(clggz + shotnoise) + (clkgz)**2.)/(2*l + 1)
    varkgzeff = ((clkk + nkk)*(clggzeff + shotnoise) + (clkgzeff)**2.)/(2*l + 1)

                  
    toret = np.array([l, clkk, nkk, clggz, clggzeff, shotnoise + l*0, clkgz, clkgzeff, varkgz, varkgzeff])

    
    order = "l, Cl_kk(unlensed, k=kappa), Noise_kk(kappa), Cl_gg(P(k, z)), Cl_gg((P(k, zeff))), Shot Noise, Cl_kg((P(k, z))), Cl_kg((P(k, z))), Varinace_kg, \
    Var_kg(Zeff) \n \
    zg = %0.2f, b1(at zg) = %0.2f, beam width = %0.2f, Delta_T = %0.2f \n \
    fsky = 1, LinearBias Evolution over dz = 0.5"%(args.zg, args.b1, args.beam, args.temp)


#    numpy.savetxt(args.outpath + "lensing_spectra/%s_%s_z%d_bw_%02d_dT_%02d.txt"%(suffix, args.survey, args.zg*100,  args.beam*10, args.temp*10), numpy.arra#y(toret).T, header = order)
    numpy.savetxt("desdropout/%s_%s_z%d_bw_%02d_dT_%02d.txt"%(suffix, args.survey, args.zg*100,  args.beam*10, args.temp*10), numpy.array(toret).T, header = order)


    #numpy.savetxt(args.outpath + "lensing_spectra/runpb_%s_z%d_bw_%02d_dT_%02d.txt"%(args.survey, args.zg*100,  args.beam*10, args.temp*10), numpy.array(toret).T, header = order)
    #numpy.savetxt(args.outpath + "lensing_spectra/lpt_%s_z%d_bw_%02d_dT_%02d.txt"%(args.survey, args.zg*100,  args.beam*10, args.temp*10), numpy.array(toret).T, header = order)
    #numpy.savetxt("./lpt_%s_z%d_bw_%02d_dT_%02d.txt"%(args.survey, args.zg*100,  args.beam*10, args.temp*10), numpy.array(toret).T, header = order)


    

