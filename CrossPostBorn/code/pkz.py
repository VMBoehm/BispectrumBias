import numpy as np
import pk_pregen as PK
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from cosmology import Cosmology
db = '../data/RunPB/'
cosmo = Cosmology(pfile='../data/RunPB/pklin_RunPB.txt', M=0.292)

#bias signature is b1, b2, bs2, bn(0), alpha, sn, auto(True/False)
toybias = [2.0, 1.0, 0., 0., 0., 0.]
toynobias = [0.0, 0.0, 0., 0., 0., 0.]


#Bias fit values
zz = 2
iz = zz*100
print(zz*2-2)
bias = np.loadtxt('../data/RunPB_datafit/log_joint_fit_results.log')
print(bias.shape)
biasx = list(bias[int(2*zz-2), 1:7])
b1 = biasx[0]
print(len(biasx))
print('biasx =', biasx)
print('b1 =', b1)
klin, plin = np.loadtxt('../data/RunPB/pklin_RunPB.txt', unpack=True)


#Theory: Returns a tuple of (k, pk)
pk = PK.PkZCLEFT(zmin=zz-0.25, zmax=zz+0.25, db=db+'/PTdata/', z0=None)
pkm = PK.PkZCLEFT(zmin=0, zmax=3.5,db=db+'/PTdata/',  z0=None)
pkmmz = lambda z: pkm([z] + [0, 0, 0, 0, 0, 0], auto=False)
pkhmz = lambda z: pk([z] + biasx, auto=False)
pkhmb1z = lambda z: pk([z] + [b1], auto=False)
#pkmmz = lambda z: pk([z] + toynobias, auto=False)


#Data
nbody = np.loadtxt(db+"hm_z%03d.pkr"%iz).T
nbody[1]/=(nbody[0]**3/2/np.pi**2)

mbody = np.loadtxt(db+"mm_z%03d.pkr"%iz).T
mbody[1]/=(mbody[0]**3/2/np.pi**2)


#Fig to save, diagnostics
fig, ax = plt.subplots(1, 2, figsize=(9,4))
axis=ax[0]
axis.plot(*pkhmz(z=zz), label='Cross')
axis.plot(*pkhmb1z(z=zz), label='Cross-b1only')
axis.plot(*pkmmz(z=zz), '--', label='Matter')
axis.plot(nbody[0], nbody[1], '.', label='Cross')
axis.plot(mbody[0], mbody[1], '.', label='Matter')
axis.loglog()
axis.legend()
axis=ax[1]
axis.plot(nbody[0], nbody[1]/np.interp(nbody[0], *pkhmz(zz)), label='cross' , marker=".")
axis.plot(nbody[0], nbody[1]/np.interp(nbody[0], *pkhmb1z(zz)), label='cross-b1only' , marker=".")
axis.plot(mbody[0], mbody[1]/np.interp(mbody[0], *pkmmz(zz)), label='Matter', marker=".")
axis.axhline(1, color='gray')
axis.legend()
axis.set_ylim(0.7, 1.3)
axis.set_xscale('log')
plt.title('z=%0.1f'%zz)
plt.savefig('../data/RunPB_datafit/testpkfit%02d.pdf'%(zz*100))

#
fig, ax = plt.subplots()
kk = pkhmz(zz)[0]
for i, z in enumerate(np.arange(zz-0.3, zz+0.3, 0.1)):
    ax.plot(kk, pkhmz(z)[1]/pkhmz(zz)[1], 'C%d'%i, label=z)
    ax.plot(kk, pkhmz(z)[1]/pkhmb1z(zz)[1], 'C%d:'%i)
    ax.plot(kk, pkmmz(z)[1]/pkmmz(zz)[1], 'C%d--'%i)
ax.set_xscale('log')
ax.legend()
plt.savefig('../data/RunPB_datafit/testpkz%02d.pdf'%(zz*100))


#Pk table to save

pktabh, pktabhb1, pktabhb1lin = [], [], []
pktabm, pktablin = [], []
pktabh.append(kk)
pktabhb1.append(kk)
pktabm.append(kk)
pktabhb1lin.append(klin)
pktablin.append(klin)


#Correct cross spectra
header = 'bias fits are (b1, b2, bs2, bn, alpha, sn) = %s, at redshift z=%.2f\n'%(str(biasx), zz)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'

for i, z in enumerate(np.arange(zz-0.25, zz+0.25, 0.01)):
    pktabh.append(pkhmz(z)[1])
pktabh = np.array(pktabh).T
print('pktabh shape ', pktabh.shape)
np.savetxt('../data/RunPB_datafit/pkhmz_tab_z%02d-interp.txt'%(zz*100), pktabh, header=header, fmt='%0.4e')


#b1 cross spectra
header = 'Using only b1_L = %0.3f, at redshift z=%.2f\n'%(b1, zz)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'

for i, z in enumerate(np.arange(zz-0.25, zz+0.25, 0.01)):
    pktabhb1.append(pkhmb1z(z)[1])
pktabhb1 = np.array(pktabhb1).T
print('pktabhb1 shape ',pktabhb1.shape)
np.savetxt('../data/RunPB_datafit/pkhmz_tab_z%02d-b1.txt'%(zz*100), pktabhb1, header=header, fmt='%0.4e')

#b1-lin
header = 'k, b1_E*plin(z)*Dgrow(z)**2 : with b1_E = %0.2f at redshift z=%.2f\n'%(1+b1, zz)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'

for i, z in enumerate(np.arange(zz-0.25, zz+0.25, 0.01)):
    pktabhb1lin.append((1+b1)*plin*cosmo.Dgrow(z=z)**2)
pktabhb1lin = np.array(pktabhb1lin).T
print('pktabhb1lin shape ',pktabhb1lin.shape)
np.savetxt('../data/RunPB_datafit/pkhmz_tab_z%02d-b1lin.txt'%(zz*100), pktabhb1lin, header=header, fmt='%0.4e')

#lin
header = 'k, plin(z)*Dgrow(z)**2 : for z in numpy.arange(0, 3.5, 0.01)'

for i, z in enumerate(np.arange(0, 3.5, 0.01)):
    pktablin.append(plin*cosmo.Dgrow(z=z)**2)
pktablin = np.array(pktablin).T
print('pktablin shape ',pktablin.shape)
np.savetxt('../data/RunPB_datafit/pkmmz_tab_z00-35-lin.txt', pktablin, header=header, fmt='%0.4e')

#Matter
header = 'k, pk(z) : for z in numpy.arange(0, 3.5, 0.01)'

for i, z in enumerate(np.arange(0, 3.5, 0.01)):
    pktabm.append(pkmmz(z)[1])
pktabm = np.array(pktabm).T
print('pktabm shape ',pktabm.shape)
np.savetxt('../data/RunPB_datafit/pkmmz_tab_z00-35-interp.txt', pktabm, header=header, fmt='%0.4e')






