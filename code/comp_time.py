import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding")
warnings.filterwarnings("ignore", message="Embedding a total of 2 separate connected components using meta-embedding")

from GRMP import GRMP
from sklearn.manifold import SpectralEmbedding, TSNE, MDS
from sklearn.decomposition import PCA
from lamp import Lamp
import umap as UMAP

from util import getConf

import time

from matplotlib import rcParams


#%% Loading data

name = 'gaussians'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

conf = getConf(name)

n_times = 10

apply_se = False
apply_pca = False
apply_lamp = False
apply_tsne = False
apply_umap = False

title = False

#%% Application of the methods

times_grmp = np.zeros((n_times,))
times_lamp = np.zeros((n_times,))
times_pca = np.zeros((n_times,))
times_tsne = np.zeros((n_times,))
times_umap = np.zeros((n_times,))
times_se = np.zeros((n_times,))

for k in range(n_times):
    # GRMP
    start = time.time()
    
    grmp = GRMP()
    
    X_grmp = grmp.fit_transform(A)
    
    times_grmp[k] = time.time()-start
    
    
    if apply_se:
        # Spectral Embedding
        start = time.time()
        
        se = SpectralEmbedding(n_components=2)
        
        X_se = se.fit_transform(A)
        
        times_se[k] = time.time()-start
    
    if apply_pca:
        # PCA
        start = time.time()
        
        pca = PCA(n_components=2)
        
        X_pca = pca.fit_transform(A)
        
        times_pca[k] = time.time()-start
    
    if apply_lamp:
        # LAMP
        start = time.time()
        
        n = A.shape[0]
        
        sample_size = int(conf.percentage_of_control_points * n)
        samples = np.random.randint(low=0,high=n,size=(sample_size,))
        
        ctp_mds = MDS(n_components=2)
        
        ctp_samples = ctp_mds.fit_transform(A[samples]-np.average(A[samples]))
        ctp_samples = np.hstack((ctp_samples,samples.reshape(sample_size,1)))
        
        #start = time.time()
        
        lamp_proj = Lamp(A, ctp_samples)
        
        X_lamp = lamp_proj.fit()
        
        times_lamp[k] = time.time()-start
           
    if apply_tsne:
        # t-SNE
        start = time.time()
        
        tsne = TSNE(n_components=2)
        
        X_tsne = tsne.fit_transform(A)
        
        times_tsne[k] = time.time()-start
    
    if apply_umap:
        # UMAP
        start = time.time()
        
        umap = UMAP.UMAP()
        
        X_umap = umap.fit_transform(A)
        
        times_umap[k] = time.time()-start


#%% Displaying the results

labelsize = 11
    
positions = [i for i in range(1,7)]

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

bp_facecolor = 'w'

plt.figure()
if title:
    plt.title(name)
    plt.ylabel('Time (in seconds)')
bp = plt.boxplot([times_grmp, times_se, times_pca, times_lamp, times_tsne, times_umap],
                 positions=positions, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(bp_facecolor)
plt.xticks(positions, ['GRMP', 'SE', 'PCA', 'LAMP', 'TSNE', 'UMAP'],
           rotation=45)
#plt.savefig('results/'+name+'/'+name+'_time.pdf', bbox_inches='tight')
plt.show()




#%% Saving data

average_time_grmp = np.mean(times_grmp)
average_time_se = np.mean(times_se)
average_time_pca = np.mean(times_pca)
average_time_lamp = np.mean(times_lamp)
average_time_tsne = np.mean(times_tsne)
average_time_umap = np.mean(times_umap)

fl = open('results/'+name+'/'+name+'_time.txt', 'w' )
fl.write('Number of runs = ' + str(n_times) + '\n')

fl.write('\n')

fl.write('Average time:\n')

fl.write('\n')

methods = ['GRMP', 'SE','PCA','LAMP','TSNE','UMAP']

averages = []

for method in methods:
    fl.write(method)
    
    if method == methods[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

averages = [average_time_grmp, average_time_se, average_time_pca,
            average_time_lamp, average_time_tsne, average_time_umap]

for average in averages:
    fl.write('%.4f' % average)
    
    if average == averages[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

fl.write('\n')

fl.close()

print("Average time GRMP: %.4f" % average_time_grmp)
