import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D; Axes3D

from GRMP import GRMP

from util import getConf, stress

import time as tm
from sklearn import metrics


#%% Loading data

name = 'digits'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

color = True

disp_silhouette = True
disp_stress = True
disp_time = True

n, m = A.shape

#%% Application of GRMP

conf = getConf(name)

start = tm.time()

grmp = GRMP(neighbors=10)
#grmp = GRMP(initial='grid')
#grmp.update_radius(2)

if name == 'iris':
    grmp.update_neighbors(neighbors=3)

X_grmp = grmp.fit_transform(A)

x_grmp = X_grmp[:,0]
y_grmp = X_grmp[:,1]

#%% Calculating metrics

if disp_time:
    time = tm.time()-start

if disp_silhouette:
    silhouette = metrics.silhouette_score(X_grmp, labels)

if disp_stress:
    stress = stress(A, X_grmp)

#%% Displaying the results

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
ax = fig.gca()
if color:
    scatter = ax.scatter(x_grmp, y_grmp, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
else:
    scatter = ax.scatter(x_grmp, y_grmp)
#plt.legend(scatter.legend_elements()[0], labels.astype(int))
ax.set_xticks([])
ax.set_yticks([])
xmin, _ = plt.xlim()
_ , ymax = plt.ylim()
fig.tight_layout()
if disp_time:
    plt.text(xmin, 0.92*ymax, '%.4f s' % time)
if disp_silhouette:
    plt.text(xmin, 0.85*ymax, '%.4f' % silhouette, color='b')
if disp_stress:
    plt.text(xmin, 0.77*ymax, '%.4f' % stress, color='r')
if name == 'outliers':
    plt.axis('equal')
plt.savefig('results/'+name+'/'+name+'_n_neighbors.pdf', bbox_inches='tight')
plt.show()

