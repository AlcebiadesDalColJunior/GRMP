import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D; Axes3D


#%% Loading data

name = 'swiss_roll'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

n, m = A.shape


#%% Displaying dataset three-dimensional visualization

n_labels = len(set(labels))

if n_labels <= 5:
    vmin = 0
    vmax = 8
    cmap = plt.get_cmap('Set1')
elif n_labels <= 12:
    vmin = 0
    vmax = 11
    cmap = plt.get_cmap('Paired')
elif n_labels <= 20:
    vmin = 0
    vmax = 19
    cmap = plt.get_cmap('tab20')
elif n_labels > 50:
    vmin = None
    vmax = None
    cmap = plt.cm.Spectral

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_adjustable("box")
ax.scatter(A[:,0], A[:,1], A[:,2], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
ax.dist = 6
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.gcf().set_size_inches((4,4))
#plt.savefig('results/'+name+'/'+name+'_original.pdf', bbox_inches='tight')
plt.savefig('results/'+name+'/'+name+'_original.eps', bbox_inches='tight')
plt.show()
