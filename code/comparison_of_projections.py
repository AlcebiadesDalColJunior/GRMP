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

from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D; Axes3D


#%% Loading data

name = 'spheres'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

conf = getConf(name)

apply_se = True
apply_pca = True
apply_lamp = True
apply_tsne = True
apply_umap = True

title = False
color = True

aspect_equal_se = False
aspect_equal_pca = False
aspect_equal_lamp = False


#%% Application of the methods

grmp = GRMP()
#grmp = GRMP(radius=0.4)

X_grmp = grmp.fit_transform(A)

x_grmp = X_grmp[:,0]
y_grmp = X_grmp[:,1]


if apply_se:
    # Spectral Embedding
    se = SpectralEmbedding(n_components=2)
    
    X_se = se.fit_transform(A)
    
    X_se = scale(X_se)
    
    x_se = X_se[:,0]
    y_se = X_se[:,1]

if apply_pca:
    # PCA
    pca = PCA(n_components=2)
    
    X_pca = pca.fit_transform(A)
    
    x_pca = X_pca[:,0]
    y_pca = X_pca[:,1]


if apply_lamp:
    # LAMP
    n = A.shape[0]
    
    sample_size = int(conf.percentage_of_control_points * n)
    
    if (False):
        unique_labels = set(labels.astype(int))
        n_labels = len(unique_labels)
        
        n_sample_per_class = int(sample_size / n_labels)
        
        samples = np.array([], dtype=np.int)
        for label in unique_labels:
            positions = np.where(labels == label)[0]
            
            sample = np.random.choice(positions, (n_sample_per_class,))
            
            samples = np.append(samples, sample)
    else:
        samples = np.random.randint(low=0,high=n,size=(sample_size,))
    
    ctp_mds = MDS(n_components=2)
    
    ctp_samples = ctp_mds.fit_transform(A[samples]-np.average(A[samples]))
    ctp_samples = np.hstack((ctp_samples,samples.reshape(sample_size,1)))
    
    lamp_proj = Lamp(A, ctp_samples)
    
    X_lamp = lamp_proj.fit()
    
    x_lamp = X_lamp[:,0]
    y_lamp = X_lamp[:,1]


if apply_tsne:
    # t-SNE
    tsne = TSNE(n_components=2)
    
    X_tsne = tsne.fit_transform(A)
    
    x_tsne = X_tsne[:,0]
    y_tsne = X_tsne[:,1]


if apply_umap:
    # UMAP
    umap = UMAP.UMAP()
    
    X_umap = umap.fit_transform(A)
    
    x_umap = X_umap[:,0]
    y_umap = X_umap[:,1]


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
if title:
    plt.title('GRMP')
if color:
    scatter = ax.scatter(x_grmp, y_grmp, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
else:
    scatter = ax.scatter(x_grmp, y_grmp)
#plt.legend(scatter.legend_elements()[0], labels.astype(int))
ax.set_xticks([])
ax.set_yticks([])
#ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('results/'+name+'/'+name+'_grmp.pdf', bbox_inches='tight')
plt.show()

if apply_se:
    fig = plt.figure()
    ax = fig.gca()
    if title:
        plt.title('SpectralEmbedding')
    if color:
        ax.scatter(x_se, y_se, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(x_se, y_se)
    ax.set_xticks([])
    ax.set_yticks([])
    if aspect_equal_se:
        ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_se.pdf', bbox_inches='tight')
    plt.show()

if apply_pca:
    fig = plt.figure()
    ax = fig.gca()
    if title:
        plt.title('PCA')
    if color:
        ax.scatter(x_pca, y_pca, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(x_pca, y_pca)
    ax.set_xticks([])
    ax.set_yticks([])
    if aspect_equal_pca:
        ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_pca.pdf', bbox_inches='tight')
    plt.show()

if apply_lamp:
    fig = plt.figure()
    ax = plt.gca()
    if title:
        plt.title('LAMP')
    if color:
        ax.scatter(x_lamp, y_lamp, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(x_lamp, y_lamp)
    ax.set_xticks([])
    ax.set_yticks([])
    if aspect_equal_lamp:
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_lamp.pdf', bbox_inches='tight')
    plt.show()

if apply_tsne:
    fig = plt.figure()
    ax = fig.gca()
    if title:
        plt.title('t-SNE')
    if color:
        ax.scatter(x_tsne, y_tsne, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(x_tsne, y_tsne)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_tsne.pdf', bbox_inches='tight')
    plt.show()

if apply_umap:
    fig = plt.figure()
    ax = plt.gca()
    if title:
        plt.title('UMAP')
    if color:
        ax.scatter(x_umap, y_umap, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(x_umap, y_umap)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_umap.pdf', bbox_inches='tight')
    plt.show()
