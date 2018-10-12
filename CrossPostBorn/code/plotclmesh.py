import numpy as np
import numpy
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.colors import LogNorm


# strng = 'baseline'
### Read in class output file and plot it folr the given column

strng = 'z300z320'
fname = '/global/u1/c/chmodi/Programs/class_public/pborncross/inifiles/class_%s_parameters.ini'%strng


with open(fname) as f:
    for line in f.readlines():
        if 'selection_mean' in line:zzs = line
    else: pass

tmp = zzs[zzs.find('=')+1:-1]
zzs = np.array([float(i) for i in tmp.split(',')])
print(zzs)
nz = zzs.size
l = np.loadtxt('/global/u1/c/chmodi/Programs/class_public/pborncross/inifiles/class_%s_cl.dat'%strng)[:, 0]

##

fig, axar = plt.subplots(nz, nz, figsize = (35, 35), sharex=True, sharey=True)
lss = ['-', '--', ':', '-.']
lws = [1, 1.5, 1.8, 2]

# for ii, strng in enumerate(['def5', 'test1', 'test2']):
# for ii, strng in enumerate(['baseline', 'baseline2', 'test22', 'test12']):
# for ii, strng in enumerate(['baseline', 'baseline2']):
# for ii, strng in enumerate(['z020z040', 'z030z050', 'z300z320']):
for ii, strng in enumerate(['z300z320']):
    cl = np.loadtxt('/global/u1/c/chmodi/Programs/class_public/pborncross/inifiles/class_%s_cl.dat'%strng)[:, 1:].T
    clmesh = np.zeros((nz, nz, l.size))
    count=0
    for i in range(nz):
        for j in range(i, nz):
            clmesh[i, j, :] = cl[count]
            clmesh[j, i, :] = cl[count]
    #         print(i, j, count)
            count +=1

    for i in range(nz):
        for j in range(nz):
            ax = axar[i, j]
#             ax[i, j].plot(l, abs(clmesh[i, j]), 'C%d%s'%(j%7,lss[ii]), label='%0.2f-%0.2f'%(zzs[i],zzs[j]))
            ax.plot(l, abs(clmesh[i, j]), 'C%d%s'%(j%7,lss[ii]), label=strng, lw=lws[ii])
            ax.set_title('%0.3f-%0.3f'%(zzs[i],zzs[j]))
            ax.semilogy()
            ax.legend()
            ax.grid()

for ii, strng in enumerate(['z300z320v2']):
    cl = np.loadtxt('/global/u1/c/chmodi/Programs/class_public/pborncross/inifiles2/class_%s_cl.dat'%strng)[:, 1:].T
    clmesh = np.zeros((nz, nz, l.size))
    count=0
    for i in range(nz):
        for j in range(i, nz):
            clmesh[i, j, :] = cl[count]
            clmesh[j, i, :] = cl[count]
    #         print(i, j, count)
            count +=1

    for i in range(nz):
        for j in range(nz):
            ax = axar[i, j]
#             ax[i, j].plot(l, abs(clmesh[i, j]), 'C%d%s'%(j%7,lss[ii]), label='%0.2f-%0.2f'%(zzs[i],zzs[j]))
            ax.plot(l, abs(clmesh[i, j]), 'C%d%s'%(j%7,lss[ii]), label=strng, lw=lws[ii], ls="--")
            ax.set_title('%0.3f-%0.3f'%(zzs[i],zzs[j]))
            ax.semilogy()
            ax.legend()
            ax.grid()
#             ax.set_xlim(0, 1000)

plt.savefig('./clmesh.png')
