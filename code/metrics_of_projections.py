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

from sklearn import metrics

from matplotlib import rcParams


#%% Loading data

name = 'digits'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

conf = getConf(name)

n_times = 10

apply_se = True
apply_pca = True
apply_lamp = True
apply_tsne = True
apply_umap = True

title = False

#%% Application of the methods

silhouettes_grmp = np.zeros((n_times,))
silhouettes_lamp = np.zeros((n_times,))
silhouettes_pca = np.zeros((n_times,))
silhouettes_tsne = np.zeros((n_times,))
silhouettes_umap = np.zeros((n_times,))
silhouettes_se = np.zeros((n_times,))

for k in range(n_times):
    # GRMP
    grmp = GRMP()
    
    X_grmp = grmp.fit_transform(A)
    
    silhouettes_grmp[k] = metrics.silhouette_score(X_grmp, labels)
    
    
    if apply_se:
        # Spectral Embedding
        se = SpectralEmbedding(n_components=2)
        
        X_se = se.fit_transform(A)
        
        silhouettes_se[k] = metrics.silhouette_score(X_se, labels)
    
    if apply_pca:
        # PCA
        pca = PCA(n_components=2)
        
        X_pca = pca.fit_transform(A)
        
        silhouettes_pca[k] = metrics.silhouette_score(X_pca, labels)
    
    if apply_lamp:
        # LAMP
        n = A.shape[0]
        
        sample_size = int(conf.percentage_of_control_points * n)
        samples = np.random.randint(low=0,high=n,size=(sample_size,))
        
        ctp_mds = MDS(n_components=2)
        
        ctp_samples = ctp_mds.fit_transform(A[samples]-np.average(A[samples]))
        ctp_samples = np.hstack((ctp_samples,samples.reshape(sample_size,1)))
        
        lamp_proj = Lamp(A, ctp_samples)
        
        X_lamp = lamp_proj.fit()
        
        silhouettes_lamp[k] = metrics.silhouette_score(X_lamp, labels)
    
    if apply_tsne:
        # t-SNE
        tsne = TSNE(n_components=2)
        
        X_tsne = tsne.fit_transform(A)
        
        silhouettes_tsne[k] = metrics.silhouette_score(X_tsne, labels)
    
    if apply_umap:
        # UMAP
        umap = UMAP.UMAP()
        
        X_umap = umap.fit_transform(A)
        
        silhouettes_umap[k] = metrics.silhouette_score(X_umap, labels)


#%% Displaying the results

labelsize = 11
    
positions = [i for i in range(1,7)]

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize

bp_facecolor = 'blue'

medianprops = dict(color="black")

plt.figure()
if title:
    plt.title(name)
    plt.ylabel('Silhouette')
bp = plt.boxplot([silhouettes_grmp, silhouettes_se, silhouettes_pca,
                  silhouettes_lamp, silhouettes_tsne, silhouettes_umap],
                  medianprops=medianprops, positions=positions, showfliers=False,
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(bp_facecolor)
plt.xticks(positions, ['GRMP', 'SE', 'PCA', 'LAMP', 'TSNE', 'UMAP'],
           rotation=45)
#plt.savefig('results/'+name+'/'+name+'_silhouette.pdf', bbox_inches='tight')
plt.show()




#%% Saving data

average_silhouettes_grmp = np.mean(silhouettes_grmp)
average_silhouettes_se = np.mean(silhouettes_se)
average_silhouettes_pca = np.mean(silhouettes_pca)
average_silhouettes_lamp = np.mean(silhouettes_lamp)
average_silhouettes_tsne = np.mean(silhouettes_tsne)
average_silhouettes_umap = np.mean(silhouettes_umap)

fl = open('results/'+name+'/'+name+'_silhouette.txt', 'w' )
fl.write('Number of runs = ' + str(n_times) + '\n')

fl.write('\n')

fl.write('Average silhouette:\n')

fl.write('\n')

methods = ['GRMP', 'SE','PCA','LAMP','TSNE','UMAP']

averages = []

for method in methods:
    fl.write(method)
    
    if method == methods[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

averages = [average_silhouettes_grmp, average_silhouettes_se, average_silhouettes_pca,
            average_silhouettes_lamp, average_silhouettes_tsne, average_silhouettes_umap]

for average in averages:
    fl.write('%.4f' % average)
    
    if average == averages[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

fl.write('\n')

fl.close()

print("Average silhouette GRMP: %.4f" % average_silhouettes_grmp)
