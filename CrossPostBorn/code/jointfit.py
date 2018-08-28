import numpy as np
import pk_pregen as PK
import scipy.optimize as opt
import datetime
import sys

# Where the data live.

db = '../data/RunPB/'


class GlobalVariables:
    dat_kk = None
    dat_dx = None
    dat_da = None
    chi2wt = None
    snerr  = None
    lpt    = None
    kfit   = None


G = GlobalVariables()


def chi2(p):
    """   
    The chi^2 given the parameters, p.
    The behavior is dictated by the length of p.
    """
    # Evaluate the theory.
    if len(p)==8:
        b1,b2,bs,bn,alpha1,sn1,alpha2,sn2=p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
        #bn=0
        sn1=0
    else:
        raise RuntimeError("Unknown p in chi2.")
    # Compute the theory
    kk,px=G.lpt([b1,b2,bs,bn,alpha1,sn1],auto=False)
    kk,pa=G.lpt([b1,b2,bs,bn,alpha2,sn2],auto=True)
    delx =kk**3*px/(2*np.pi**2)
    dela =kk**3*pa/(2*np.pi**2)
    # Now interpolate the theory onto the N-body samples.
    thy_x = np.interp(G.dat_kk,kk,delx)
    thy_a = np.interp(G.dat_kk,kk,dela)
    # Set the weights
    wt_x  = np.exp(-(G.dat_kk/G.kfit[0])**2)
    wt_x  = 1.0-G.dat_kk/G.kfit[0]
    wt_x[G.dat_kk<0.05]=0
    wt_x[G.dat_kk>G.kfit[0]]=0
    wt_a  = np.exp(-(G.dat_kk/G.kfit[1])**2)
    wt_a  = 1.0-G.dat_kk/G.kfit[1]
    wt_a[G.dat_kk<0.05]=0
    wt_a[G.dat_kk>G.kfit[1]]=0
    # and compute chi^2.
    c2 = G.chi2wt[0] * np.sum( (G.dat_dx-thy_x)**2*wt_x ) +\
         G.chi2wt[1] * np.sum( (G.dat_da-thy_a)**2*wt_a )
    # and add a prior on sn1 and sn2 to drive them to zero.
    # ADD prior on bn as well
    p2 = 1.*p[3]**2 + 1.*p[5]**2 + 0.5*(p[7]/G.snerr)**2 #+ 100.*p[2]**2.
    return(c2+p2)
    #



def fit_example(verbose=False, lo = False):
    """    
    An example fitter.
    """
    # Clear out the past fits.
    pname = ["b1","b2","bs","bn","alpha1","sn1","alpha2","sn2"]
    fname = "../data/RunPB_datafit/log_joint_fit_results.log"
    if lo: fname = "../data/RunPB_datafit/log_joint_fit_results_lo.log"
    ff= open(fname,"w")
    ff.write("# Results of joint fit to HH and HM spectra.\n")
    ff.write("# "+str(datetime.datetime.now()).split('.')[0]+'\n')
    st= "# iz"
    for p in pname:
        st += " %12s"%p
    ff.write(st+" %12s\n"%"chi2")
    ff.close()
    # Read the shot-noise values to subtract.
    if lo:
        sn = 1.0/np.loadtxt(db+"nbar.txt",usecols=(1,))
    else:
        sn = 1.0/np.loadtxt(db+"nbar.txt",usecols=(2,))
    # Some starting values for the fitter
    b1 = {100: 0.42, 150: 0.85, 200: 1.60, 250:2.20, 300: 3.10}
    kmax_x = [0.30,0.30,0.40,0.40,0.40]
    kmax_a = [0.35,0.25,0.30,0.25,0.25]
    if lo:
        b1 = {100: 0.3, 150: 0.5, 200: 1.2, 250:1.80, 300: 2.50}
        kmax_x = [0.45,0.30,0.60,0.40,0.40]
        kmax_a = [0.40,0.25,0.50,0.25,0.40]
    for ii,iz in enumerate([100,150,200,250,300]):
        #G.lpt    = PK.PkCLEFT(db+"PTdata/ps00_hh_RunPB_46_z%03d.dat"%iz)
        G.lpt    = PK.PkZCLEFT(zmin=iz/100-0.25, zmax=iz/100.+0.25, z0=iz/100., db=db+"PTdata/")
        G.kfit   = [1.0,1.0]
        G.chi2wt = [1.0,1.0]
        G.snerr  = 0.02*sn[ii]
        print(ii,iz,G.snerr)
        #
        nbody = np.loadtxt(db+"hm_z%03d.pkr"%iz)
        if lo:
            nbody = np.loadtxt(db+"hm_z%03d_lo.pkr"%iz)
        G.dat_kk = nbody[5:,0]
        G.dat_dx = nbody[5:,1]
        G.kfit[0]= kmax_x[ii]
        nbody = np.loadtxt(db+"hh_z%03d.pkr"%iz)
        if lo:
            nbody = np.loadtxt(db+"hh_z%03d_lo.pkr"%iz)
        G.dat_da = nbody[5:,1] - sn[ii]*nbody[5:,0]**3/2/np.pi**2
        G.kfit[1]= kmax_a[ii]
        # Use the Nelder-Mead method, because the other methods start
        # using really odd values of the parameters.
        pars    = [b1[iz],0.0,0.0,0.0,0.0,1.0,0.0,1.0]
        solvopt = {'disp':True,'maxiter':5000,'maxfev':15000,'xtol':1e-4}
        res = opt.minimize(chi2,pars,method='nelder-mead',options=solvopt)
        # Optionally print the best fit and fit information.
        #print(res)
        p = res.x
        p[3],p[5],p[7]=0.0,0.0,0	# Zero the shot noise terms. And bn
        ff= open(fname,"a")
        st= " %03d"%iz
        for x in p:
            st += " %12.4e"%x
        ff.write(st+" %12.4e\n"%res.fun)
        ff.close()
        if verbose:
            print("\nMatch @ iz=",iz,"\n")
            kk,px=G.lpt([p[0],p[1],p[2],p[3],p[4],p[5]],auto=False)
            kk,pa=G.lpt([p[0],p[1],p[2],p[3],p[6],p[7]],auto=True)
            delx =kk**3*px/(2*np.pi**2)
            dela =kk**3*pa/(2*np.pi**2)
            # Now interpolate the theory onto the N-body samples.
            thy_x = np.interp(G.dat_kk,kk,delx)
            thy_a = np.interp(G.dat_kk,kk,dela)
            for i in range(G.dat_kk.shape[0]):
                print("%3d %8.5f %12.4e %12.4e %8.3f %12.4e %12.4e %8.3f"%\
                     (i,G.dat_kk[i],G.dat_dx[i],thy_x[i],\
                     100*(G.dat_dx[i]-thy_x[i])/G.dat_dx[i],\
                     G.dat_da[i],thy_a[i],\
                     100*(G.dat_da[i]-thy_a[i])/G.dat_da[i]))
    #


if __name__=="__main__":
    print(sys.argv)
    try:
        if sys.argv[1] == 'lo':
            print('For lo mass halos')
            lo=True        
    except:
        print('For 10^12M halos')
        lo=False
    #fit_example(verbose=False,lo=lo)
    #
    fit_example(verbose=False,lo=False)
    #
