import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import sys, importlib
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.interpolate import interp1d
#
import pk_pregen as PK
import lensingZ, tools
from cosmology import Cosmology

cosmo = Cosmology(pfile='../data/RunPB/pklin_RunPB.txt', M=0.292)
z0, dz = 2, 0.25
bias = np.loadtxt('../data/RunPB_datafit/log_joint_fit_results.log')
print(bias.shape)
biasx = list(bias[int(2*z0-2), 1:7])
b1 = biasx[0]


#matter
pkmm = np.loadtxt('../data/RunPB_datafit/pkmmz_tab_z00-35-interp.txt')
kk = pkmm[:, 0]
pkmm = pkmm[:, 1:]
zzm = np.arange(0,3.5,0.01)
ipkmmz = interp1d(zzm, pkmm , fill_value=0, bounds_error=False)

#matter lin 
pklin = np.loadtxt('../data/RunPB_datafit/pkmmz_tab_z00-35-lin.txt')
klin = pklin[:, 0]
plin = pklin[:, 1:]
ipklinz = interp1d(zzm, plin , fill_value=0, bounds_error=False)

#halos
pkhm = np.loadtxt('../data/RunPB_datafit/pkhmz_tab_z%02d-interp.txt'%(z0*100))[:, 1:]
zzh = np.arange(z0-dz, z0+dz, 0.01)
ipkhmz = interp1d(zzh, pkhm , fill_value=0, bounds_error=False)

#halo b1
pkhmb1 = np.loadtxt('../data/RunPB_datafit/pkhmz_tab_z%02d-b1.txt'%(z0*100))[:, 1:]
ipkhmb1z = interp1d(zzh, pkhmb1 , fill_value=0, bounds_error=False)

#halo b1-lin
pkhmb1lin = np.loadtxt('../data/RunPB_datafit/pkhmz_tab_z%02d-b1lin.txt'%(z0*100))[:, 1:]
ipkhmb1linz = interp1d(zzh, pkhmb1lin , fill_value=0, bounds_error=False)

#
ikpkmmz = lambda z: (kk, ipkmmz(z))
ikpkhmz = lambda z: (kk, ipkhmz(z))
ikpklinz = lambda z: (klin, ipklinz(z))
ikpkhmb1z = lambda z: (kk, ipkhmb1z(z))
ikpkhmb1linz = lambda z: (klin, ipkhmb1linz(z))

#Diagnostic
fig, axis = plt.subplots(1,2, figsize=(9, 4))
ax=axis[0]
ax.plot(*ikpkmmz(z0), 'C0', label='Matter, 1 loop')
ax.plot(*ikpklinz(z0), 'C1--', label='Linear')
ax.plot(*ikpkhmz(z0), 'C2', label='Cross All bias')
ax.plot(*ikpkhmb1z(z0), 'C3--', label='b1, 1 loop')
ax.plot(*ikpkhmb1linz(z0), 'C4:', label='b1lin')
ax.grid(which='both', lw=0.1, color='gray')
ax.legend()
ax.loglog()

ax=axis[1]
ax.plot(kk, ikpkhmb1z(z0)[1]/interpolate(*ikpkhmb1linz(z0))(kk))
#ax.plot(kk, ikpkhmb1z(z0)[1]/interpolate(*ikpkhmb1linz(z0))(kk)/b1, 'r--')
ax.axhline(1, lw=0.5)
#ax.plot(kk, pkmm[:, 200], 'C0:', label='Matter, 1 loop', lw=3)
#ax.plot(klin, plin[:, 200], 'C1:', label='Linear', lw=3)
#ax.plot(kk, pkhm[:, 25], 'C2:', label='Cross All bias', lw=3)
#ax.plot(kk, pkhmb1[:, 25], 'C3:', label='b1, 1 loop', lw=3)
#ax.plot(klin, pkhmb1lin[:, 25], 'C4:', label='b1lin', lw=3)
ax.grid(which='both', lw=0.1, color='gray')
ax.legend()
ax.set_xscale('log')
plt.savefig('testpkz.pdf')


#Pborn
lmin, lmax = 1e-2, 3000


for lmin in [1e-2, 1e-4]:
    for lmax in [3000, 5000, 10000]:

    
        print(lmin, lmax)
        lenz = lensingZ.LensingZ(z0, dndz=tools.DnDz().lsst, cosmo=cosmo, dz=dz, l=np.logspace(np.log10(lmin), np.log10(lmax), 5001), purelimber=True)

        ell = lenz.l
        print('Signal')
        clkg = lenz.clz(ikpkhmz, auto=False)
        print('PostBorn')
        pborn = lenz.clkg31d(ikpkhmz, ikpkmmz)

        #pbornlin = lenz.clkg31d(ikpkhmz, ikpklinz)
        #pbornb1lin = lenz.clkg31d(ikpkhmb1z, ikpklinz)
        pbornb1linlin = lenz.clkg31d(ikpkhmb1linz, ikpklinz)

        print('Figures')
        fig, ax = plt.subplots(1,1, figsize=(5, 4))

        #axis=ax[0]
        axis=ax
        axis.plot(ell, clkg, 'C0', label='Signal')
        axis.plot(ell, pborn, 'C1', label='PostBorn', lw=2, alpha=1)
        #axis.plot(ell, pbornlin, 'C2--', label='allbias-lin', lw=2, alpha=1)
        #axis.plot(ell, pbornb1lin, 'C3:', label='b1(1loop)-lin', lw=3, alpha=0.8)
        axis.plot(ell, pbornb1linlin, 'C4-.', label='b1lin-lin', lw=3, alpha=0.5)
        #axis.plot(ell, -1*pborn, 'C1--', label='PostBorn')
        axis.loglog()
        axis.legend()
        axis.set_ylim(1e-13, 1e-6)
        if lmax > 5000: axis.set_xlim(10, lmax)
        else: axis.set_xlim(10, 5000)
        axis.grid(which='both', lw=0.1, color='gray')
        axis.axvline(2000, color='r', lw=0.5, ls="--")
        axis.axhline(5e-10, color='r', lw=0.5, ls="--")

        #axis=ax[1]
        #axis.plot(ell, abs(pborn)/clkg, 'C1', label='Ratio')
        #axis.plot(ell, abs(pbornb1linlin)/clkg, 'C4--', label='Ratio')
        #axis.loglog()
        #axis.legend()
        #axis.set_xlim(10, 5000)
        #axis.grid(which='both', lw=0.1, color='gray')
        plt.suptitle('lmin = %0.2e, lmax = %0.2f, z0=%.1f, dz=%0.2f, b1=%.2f'%(ell[0], ell[-1], z0, dz, b1))
        plt.savefig('../results/plots/epic_fail/pborn-lmin%02d-lmax%02d_z%02d.pdf'%(-1*np.log10(lmin), lmax/100, z0*100))
