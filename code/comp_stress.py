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

from util import getConf, stress

from matplotlib import rcParams


#%% Loading data

name = 'iris'

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

stresses_grmp = np.zeros((n_times,))
stresses_lamp = np.zeros((n_times,))
stresses_pca = np.zeros((n_times,))
stresses_tsne = np.zeros((n_times,))
stresses_umap = np.zeros((n_times,))
stresses_se = np.zeros((n_times,))

for k in range(n_times):
    # GRMP
    grmp = GRMP()
    
    X_grmp = grmp.fit_transform(A)
    
    stresses_grmp[k] = stress(A, X_grmp)
    
    
    if apply_se:
    # Spectral Embedding
        se = SpectralEmbedding(n_components=2)
        
        X_se = se.fit_transform(A)
        
        stresses_se[k] = stress(A, X_se)
    
    if apply_pca:
        # PCA
        pca = PCA(n_components=2)
        
        X_pca = pca.fit_transform(A)
        
        stresses_pca[k] = stress(A, X_pca)
    
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
        
        stresses_lamp[k] = stress(A, X_lamp)
    
    if apply_tsne:
        # t-SNE
        tsne = TSNE(n_components=2)
        
        X_tsne = tsne.fit_transform(A)
        
        stresses_tsne[k] = stress(A, X_tsne)
    
    if apply_umap:
        # UMAP
        umap = UMAP.UMAP()
        
        X_umap = umap.fit_transform(A)
        
        stresses_umap[k] = stress(A, X_umap)


#%% Displaying the results

labelsize = 11
    
positions = [i for i in range(1,7)]

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

bp_facecolor = 'red'

medianprops = dict(color="black")

plt.figure()
if title:
    plt.title(name)
    plt.ylabel('Stress')
bp = plt.boxplot([stresses_grmp, stresses_se, stresses_pca, stresses_lamp, stresses_tsne,
                  stresses_umap], positions=positions, showfliers=False,
                  medianprops=medianprops, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(bp_facecolor)
plt.xticks(positions,['GRMP','SE','PCA','LAMP','TSNE','UMAP'],rotation=45)
#plt.savefig('results/'+name+'/'+name+'_stress.pdf', bbox_inches='tight')
plt.show()




#%% Saving data

average_stress_grmp = np.mean(stresses_grmp)
average_stress_se = np.mean(stresses_se)
average_stress_pca = np.mean(stresses_pca)
average_stress_lamp = np.mean(stresses_lamp)
average_stress_tsne = np.mean(stresses_tsne)
average_stress_umap = np.mean(stresses_umap)


fl = open('results/'+name+'/'+name+'_stress.txt', 'w' )
fl.write('Number of runs = ' + str(n_times) + '\n')

fl.write('\n')

fl.write('Average stress:\n')

fl.write('\n')

methods = ['GRMP', 'SE','PCA','LAMP','TSNE','UMAP']

averages = []

for method in methods:
    fl.write(method)
    
    if method == methods[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

averages = [average_stress_grmp, average_stress_se, average_stress_pca, average_stress_lamp,
            average_stress_tsne, average_stress_umap]

for average in averages:
    fl.write('%.4f' % average)
    
    if average == averages[-1]:
        fl.write(' \\\\ \hline \n')
    else:
        fl.write(' & ')

fl.write('\n')

fl.close()

print("Average stresse GRMP: %.4f" % average_stress_grmp)
