import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import rcParams

from sklearn.datasets import make_swiss_roll

from scipy.sparse.linalg import eigsh

from sklearn.neighbors import kneighbors_graph


#%% Main code

name = 'swiss_roll'

n_points = 1500 

A = make_swiss_roll(n_points)[0]

W = kneighbors_graph(A, 5, include_self=False)

G = nx.from_scipy_sparse_matrix(W)

L = nx.laplacian_matrix(G)

v0 = np.ones((n_points,))

labels = eigsh(L, k=2, which='SA', tol=1e-08, maxiter=1e8, v0=v0)[1][:,1]


#%% Plotting results

labelsize = 11

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

vmin = None
vmax = None
cmap = plt.cm.Spectral

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
ax.dist = 7
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.xticks([])
plt.yticks([])
plt.show()


#%% Saving data

np.savetxt('results/'+name+'.csv', A, fmt='%.6e')
np.savetxt('results/'+name+'_labels.csv', labels, fmt='%.6e')
