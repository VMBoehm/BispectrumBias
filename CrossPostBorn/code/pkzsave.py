import numpy as np
import pk_pregen as PK
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from cosmology import Cosmology
db = '../data/RunPB/'
cosmo = Cosmology(pfile='../data/RunPB/pklin_RunPB.txt', M=0.292)
klin, plin = np.loadtxt('../data/RunPB/pklin_RunPB.txt', unpack=True)

#bias signature is b1, b2, bs2, bn(0), alpha, sn, auto(True/False)
toybias = [2.0, 1.0, 0., 0., 0., 0.]
toynobias = [0.0, 0.0, 0., 0., 0., 0.]


#Bias fit values
z0 = 2
iz = z0*100
print(z0*2-2)
bias = np.loadtxt('../data/RunPB_datafit/log_joint_fit_results.log')
print(bias.shape)
biasx = list(bias[int(2*z0-2), 1:7])
b1 = biasx[0]
b1E = 1+b1
b1Ez = lambda z: b1E*cosmo.Dgrow(z=2)/cosmo.Dgrow(z=z)
fig = plt.figure()
plt.plot(np.linspace(0, 3.5), b1Ez(np.linspace(0, 3.5)))
plt.savefig('b1z.pdf')
print(len(biasx))
print('biasx =', biasx)
print('b1 =', b1)


#Theory: Returns a tuple of (k, pk)
pk = PK.PkZCLEFT(zmin=z0-0.25, zmax=z0+0.25, db=db+'/PTdata/', z0=None)
pkm = PK.PkZCLEFT(zmin=0, zmax=3.5,db=db+'/PTdata/',  z0=None)
kk = pk([z0])[0]
#
pkmmz = lambda z: pkm([z] + [0, 0, 0, 0, 0, 0], auto=False)[1]
pkmmlinz = lambda z: plin*cosmo.Dgrow(z=z)**2
pkmmhfz = lambda z: cosmo.pkanlin(z=z)[1]
#
pkhmz = lambda z: pk([z] + biasx, auto=False)[1]
pkhmb1z = lambda z: pk([z] + [b1], auto=False)[1]
pkhmb1linz = lambda z: b1E*plin*cosmo.Dgrow(z=z)**2
pkhmb1zlinz = lambda z: b1Ez(z=z)*plin*cosmo.Dgrow(z=z)**2
pkhmb1zhfz = lambda z: cosmo.pkanlin(z=z)[1]*b1Ez(z=z)

#pkmmz = lambda z: pk([z] + toynobias, auto=False)


def savetab(k, ipkz, zmin, zmax, header, fname, dz=0.01):
    pktab = []
    pktab.append(k)
    for i, z in enumerate(np.arange(zmin, zmax, dz)):
        pktab.append(ipkz(z))
    pktab = np.array(pktab).T
    print('For fname = %s, pktab shape '%fname, pktab.shape)
    np.savetxt('../data/RunPB_datafit/%s'%fname, pktab, header=header, fmt='%0.4e')
    


#Correct cross spectra from CLEFT
header = 'bias fits are (b1, b2, bs2, bn, alpha, sn) = %s, at redshift z=%.2f\n'%(str(biasx), z0)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'
fname = 'pkhmz_tab_z%02d-cleft.txt'%(z0*100)
savetab(kk, pkhmz, z0-0.25, z0+0.25, header=header, fname=fname)


#b1 cross spectra from CLEFT
header = 'Using only b1_L = %0.3f in CLEFT, at redshift z=%.2f\n'%(b1, z0)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'
fname = 'pkhmz_tab_z%02d-b1cleft.txt'%(z0*100)
savetab(kk, pkhmb1z, z0-0.25, z0+0.25, header=header, fname=fname)


#b1-lin
header = 'k, b1_E*plin(z)*Dgrow(z)**2 : with b1_E = %0.2f at redshift z=%.2f\n'%(b1, z0)
header += 'k, pk(z) : for z in numpy.arange(z-0.25, z+0.25, 0.01)'
fname = 'pkhmz_tab_z%02d-b1lin.txt'%(z0*100)
savetab(klin, pkhmb1linz, z0-0.25, z0+0.25, header=header, fname=fname)

#b1(z)-lin
header = 'k, pk(z)=b1E(z)*pklin(z) : for z in numpy.arange(0, 3.5, 0.01), Linear Spectra\n'
header += 'b1(z) = b1E(z=2, =%0.2f) * Dgrow(z=2, =%0.2f)/Dgrow(z=z)'%(b1E, cosmo.Dgrow(z=2))
fname = 'pkhmz_tab_z00-35-b1zlin.txt'
savetab(klin, pkhmb1zlinz, 0, 3.5, header=header, fname=fname)


#b1(z)*Halofit
header = 'k, pk(z)=b1E(z)*pkhf(z) : for z in numpy.arange(0, 3.5, 0.01), Halofit Spectra\n'
header += 'b1(z) = b1E(z=2, =%0.2f) * Dgrow(z=2, =%0.2f)/Dgrow(z=z)'%(b1E, cosmo.Dgrow(z=2))
fname = 'pkhmz_tab_z00-35-b1zhfit.txt'
savetab(klin, pkhmb1zhfz, 0, 3.5, header=header, fname=fname)


#lin
header = 'k, plin(z)*Dgrow(z)**2 : for z in numpy.arange(0, 3.5, 0.01)'
fname = 'pkmmz_tab_z00-35-lin.txt'
savetab(klin, pkmmlinz, 0, 3.5, header=header, fname=fname)

#Matter - CLEFT
header = 'k, pk(z) : for z in numpy.arange(0, 3.5, 0.01), CLEFT 1loop spectra'
fname = 'pkmmz_tab_z00-35-cleft.txt'
savetab(kk, pkmmz, 0, 3.5, header=header, fname=fname)

#Matter - Halofit
header = 'k, pk(z) : for z in numpy.arange(0, 3.5, 0.01), Halofit Spectra'
fname = 'pkmmz_tab_z00-35-hfit.txt'
savetab(klin, pkmmhfz, 0, 3.5, header=header, fname=fname)









