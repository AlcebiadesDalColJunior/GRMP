import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import rcParams

from GRMP import GRMP

from util import getConf, stress
from sklearn import metrics

#import time


#%% Loading data

name = 'digits'

apply_silhouette = True
apply_stress = True

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

conf = getConf(name)


#%% Main code

neighbors = list(range(3,12))

n_neighbors = len(neighbors)

if apply_stress:
    stresses = np.zeros((n_neighbors,))
    
    grmp = GRMP()
    
    for i,n in enumerate(neighbors):
        grmp.update_neighbors(n)
        
        X_grmp = grmp.fit_transform(A)
        
        stresses[i] = stress(A, X_grmp)

if apply_silhouette:
    silhouettes = np.zeros((n_neighbors,))
    
    grmp = GRMP()
    
    for i,n in enumerate(neighbors):
        grmp.update_neighbors(n)
        
        X_grmp = grmp.fit_transform(A)
        
        silhouettes[i] = metrics.silhouette_score(X_grmp, labels)

silhouette_max = np.max(silhouettes)

#ind = np.argmax(silhouettes)
#n_neighbors = neighbors[ind]
#
#print('Optimal number of neighbors: %i' % n_neighbors)
#print('Maximum silhouette: %.4f' % silhouette_max)

#if (False):
#    nnz_L = np.zeros((n_neighbors,))
#    times = np.zeros((n_neighbors,))
#    
#    n_times = 1
#    
#    for i,n in enumerate(neighbors):
#        ttime = 0
#        for k in range(n_times):
#            start = time.time()
#            
#            grmp = GRMP()
#            
#            grmp.update_neighbors(n)
#            
#            grmp.fit_transform(A)
#            
#            ttime += time.time()-start
#        
#        times[i] = ttime / n_times
#        
#        G = grmp.G
#        
#        L = nx.laplacian_matrix(G)
#        
#        W = nx.adjacency_matrix(G)
#        
#        plt.figure()
#        plt.imshow(W.toarray())
#        plt.show()
#        
#        #nnz_L[i] = L.getnnz()
#        nnz_L[i] = W.getnnz() / 2


#%% Plotting results

labelsize = 11

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

plt.figure()
if apply_stress:
    plt.plot(neighbors, stresses, c='r', label='stress')
if apply_silhouette:
    plt.plot(neighbors, silhouettes, c='b', label='silhouette', zorder=1)
#if (False):
#    plt.plot(neighbors, times, c='b', label='times')
#if (False):
#    plt.bar(neighbors, nnz_L)
#plt.scatter(n_neighbors, silhouette_max, c='r', zorder=2)
plt.xticks(neighbors, neighbors)
plt.legend()
#plt.xlabel('Number of nearest neighbors')
#plt.xlabel('k')
#if apply_silhouette:
#    plt.ylabel('Silhouette')
plt.grid('true')
plt.savefig('results/'+name+'/'+name+'_n_neighbors.pdf', bbox_inches='tight')
plt.show()
