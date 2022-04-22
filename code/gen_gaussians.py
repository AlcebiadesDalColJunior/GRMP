import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

from sklearn.decomposition import PCA




#%% Main code

save = False

name = 'gaussians'

apply_pca = True

n_points = 10000 #1500
dim = 25
n_gauss = 20 #4


A = np.zeros((n_points,dim))
labels = np.zeros((n_points,))

mean = np.zeros((n_gauss,dim))
for i in range(n_gauss):
    mean[i,:] = np.random.randint(2, size=dim)

#n_points_class = [50, 200, 400, 850]
#n_points_class = [40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100,
#                  500, 500, 2000, 6000]
n_points_class = [n_points//n_gauss for i in range(n_gauss)]

sup = 0
for i in range(n_gauss):
    inf = sup
    sup = inf + n_points_class[i]
    
    cov = np.diag(np.random.uniform(low=0.01, high=0.03, size=dim))
    
    A[inf:sup,:] = np.random.multivariate_normal(mean[i,:], cov, n_points_class[i])
    
    labels[inf:sup] = i





#%% Plotting results

if apply_pca:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(A)
    
    x_pca = X_pca[:,0]
    y_pca = X_pca[:,1]
    
    labelsize = 11
    
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize 
    
    vmin = 0
    vmax = 8
    cmap = plt.get_cmap('Set1')
    
    s = 40
    
    plt.figure()
    plt.scatter(x_pca, y_pca, c=labels, cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    plt.xticks([])
    plt.yticks([])
    plt.show()




#%% Saving data

if save:
    np.savetxt('datasets/'+name+'.csv', A, fmt='%.6e')
    np.savetxt('datasets/'+name+'_labels.csv', labels, fmt='%.6e')
    
    fl = open('datasets/'+name+'_specifications.txt', 'w' )
    fl.write( 'Number of points = ' + str(n_points) + '\n' )
    fl.write( 'Number of gaussians = ' + str(n_gauss) + '\n' )
    fl.write( 'Number of points per class = ' + str(n_points_class) + '\n' )
    fl.write( 'Dimension = ' + str(dim) + '\n' )
    fl.close()
