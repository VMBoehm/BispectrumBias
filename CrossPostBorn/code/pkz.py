import numpy as np
import pk_pregen as PK
import matplotlib.pyplot as plt
plt.switch_backend('agg')

db = '../data/RunPB/'


#bias signature is b1, b2, bs2, bn(0), alpha, sn, auto(True/False)
toybias = [2.0, 1.0, 0., 0., 0., 0.]
toynobias = [0.0, 0.0, 0., 0., 0., 0.]



zz = 2
iz = zz*100
print(zz*2-2)
bias = np.loadtxt('../data/RunPB_datafit/log_joint_fit_results.log')
print(bias.shape)
biasx = list(bias[int(2*zz-2), 1:7])
print(len(biasx))
print('biasx =', biasx)

#Returns a tuple of (k, pk)
#pk = PK.PkZCLEFT(zmin=1.5, zmax=2.5, z0=zz, db=db+'/PTdata/')
pk = PK.PkZCLEFT(zmin=zz-0.25, zmax=zz+0.25, db=db+'/PTdata/')
pkhmz = lambda z: pk([z] + biasx, auto=False)
pkmmz = lambda z: pk([z] + toynobias, auto=False)

nbody = np.loadtxt(db+"hm_z%03d.pkr"%iz).T
nbody[1]/=(nbody[0]**3/2/np.pi**2)

mbody = np.loadtxt(db+"mm_z%03d.pkr"%iz).T
mbody[1]/=(mbody[0]**3/2/np.pi**2)

#Fig
fig, ax = plt.subplots(1, 2, figsize=(9,4))
axis=ax[0]
axis.plot(*pkhmz(z=zz), label='Cross')
axis.plot(*pkmmz(z=zz), '--', label='Matter')
axis.plot(nbody[0], nbody[1], '.', label='Cross')
axis.plot(mbody[0], mbody[1], '.', label='Matter')
axis.loglog()
axis.legend()
axis=ax[1]
axis.plot(nbody[0], nbody[1]/np.interp(nbody[0], *pkhmz(zz)), label='cross' , marker=".")
axis.plot(mbody[0], mbody[1]/np.interp(mbody[0], *pkmmz(zz)), label='Matter', marker=".")
axis.axhline(1, color='gray')
axis.legend()
axis.set_ylim(0.7, 1.3)
axis.set_xscale('log')
plt.title('z=%0.1f'%zz)
plt.savefig('../data/RunPB_datafit/testpkfit%02d.pdf'%(zz*100))

fig, ax = plt.subplots()
kk = pkhmz(zz)[0]
for i, z in enumerate(np.arange(zz-0.3, zz+0.3, 0.1)):
    ax.plot(kk, pkhmz(z)[1]/pkhmz(zz)[1], 'C%d'%i, label=z)
    ax.plot(kk, pkmmz(z)[1]/pkmmz(zz)[1], 'C%d--'%i)
ax.set_xscale('log')
ax.legend()
plt.savefig('../data/RunPB_datafit/testpkz%02d.pdf'%(zz*100))


header = 'bias fits are (b1, b2, bs2, bn, alpha, sn) = %s, at redshift z=%.2f\n'%(str(biasx), zz)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'

pktabh, pktabm = [], []
pktabh.append(kk)
pktabm.append(kk)
for i, z in enumerate(np.arange(zz-0.25, zz+0.25, 0.01)):
    #header += '     '+ str(z)
    pktabh.append(pkhmz(z)[1])
    pktabm.append(pkmmz(z)[1])
pktabh = np.array(pktabh).T
pktabm = np.array(pktabm).T

np.savetxt('../data/RunPB_datafit/pkhmz_tab_z%02d.txt'%(zz*100), pktabh, header=header, fmt='%0.4e')
np.savetxt('../data/RunPB_datafit/pkmmz_tab_z%02d.txt'%(zz*100), pktabm, header=header, fmt='%0.4e')
