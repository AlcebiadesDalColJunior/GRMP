import matplotlib.pyplot as plt

from matplotlib import rcParams

from mpl_toolkits.mplot3d import Axes3D; Axes3D

import numpy as np




#%% Main code

save = False

name = 'spheres'

def sample_spherical(npoints, ndim=3, radius=1):
    #vec = np.random.randn(ndim, npoints)
    vec = 2 * np.random.random((ndim, npoints)) -1
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    
    return vec

#def sample_spherical(npoints, ndim=3, radius=1):    
#    x = np.zeros((npoints,))
#    y = np.zeros((npoints,))
#    z = np.zeros((npoints,))
#    
#    for i in range(npoints):
#        phi = np.pi * np.random.rand()
#        theta = 2 * np.pi * np.random.rand()
#        
#        x[i] = radius * np.sin(phi) * np.cos(theta)
#        y[i] = radius * np.sin(phi) * np.sin(theta)
#        z[i] = radius * np.cos(phi)
#    
#    return x,y,z

n_points = 1000
n_spheres = 3

x = dict()
y = dict()
z = dict()

for i in range(n_spheres):
    x[i], y[i], z[i] = sample_spherical(n_points, radius=i+1)

n = n_spheres * n_points

A = np.zeros((n,3))
labels = np.zeros((n,))

index = 0
for i in range(n_spheres):
    for j in range(n_points):
        A[index,0] = x[i][j]
        A[index,1] = y[i][j]
        A[index,2] = z[i][j]
        
        labels[index] = i
        
        index += 1




#%% Plotting results

labelsize = 11

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

vmin = 0
vmax = 8
cmap = plt.get_cmap('Set1')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
ax.dist = 7
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.xticks([])
plt.yticks([])
plt.show()




#%% Saving data

if (save): 
    np.savetxt('datasets/'+name+'.csv', A, fmt='%.6e')
    np.savetxt('datasets/'+name+'_labels.csv', labels, fmt='%.6e')
