import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D; Axes3D

from GRMP import GRMP

from util import getConf


#%% Loading data

name = 'iris'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

G_on_projection = True

color = True

n, m = A.shape

#%% Application of GRMP

conf = getConf(name)

grmp = GRMP()
# grmp = GRMP(initial='grid')
#grmp.update_radius(2)

if name == 'iris':
    grmp.update_neighbors(neighbors=3)

X_grmp = grmp.fit_transform(A)

x_grmp = X_grmp[:,0]
y_grmp = X_grmp[:,1]


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
#plt.savefig('results/'+name+'/'+name+'_grmp.pdf', bbox_inches='tight')
plt.savefig('results/'+name+'/'+name+'_grmp.eps', bbox_inches='tight')
plt.show()


x_init = grmp.x_init
y_init = grmp.y_init

G = grmp.G
graphs = grmp.graphs

for i,graph in enumerate(graphs):
    nodes = list(graph)
    
    c = labels[nodes]
    
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x_init[nodes], y_init[nodes], c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.scatter(x_init[grmp.control_nodes[i]], y_init[grmp.control_nodes[i]], c='y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    fig.tight_layout()
    #plt.savefig('results/'+name+'/'+name+'_init_'+str(i)+'.pdf', bbox_inches='tight')
    plt.savefig('results/'+name+'/'+name+'_init_'+str(i)+'.eps', bbox_inches='tight')
    plt.show()
    
    if G_on_projection:
        pos = dict()
        for node in nodes:
            pos[node] = np.array([x_init[node], y_init[node]])
        
        fig = plt.figure()
        ax = fig.gca()
        nx.draw_networkx_nodes(G.subgraph(graph), pos, node_size=60, node_color=c, cmap=cmap, vmin=vmin, vmax=vmax)
        edges = nx.draw_networkx_edges(G.subgraph(graph), pos, width=2.0, edge_color='gray')
        ax.set_aspect('equal')
        fig.tight_layout()
        #plt.savefig('results/'+name+'/'+name+'_init_G_on_projection_'+str(i)+'.pdf', bbox_inches='tight')
        plt.savefig('results/'+name+'/'+name+'_init_G_on_projection_'+str(i)+'.eps', bbox_inches='tight')
        plt.show()


x_filtered = grmp.x_filtered
y_filtered = grmp.y_filtered

for i,graph in enumerate(graphs):
    nodes = list(graph)
    
    c = labels[nodes]

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x_filtered[nodes], y_filtered[nodes], c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.scatter(x_filtered[grmp.control_nodes[i]], y_filtered[grmp.control_nodes[i]], c='y')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    #plt.savefig('results/'+name+'/'+name+'_filtered_'+str(i)+'.pdf', bbox_inches='tight')
    plt.savefig('results/'+name+'/'+name+'_filtered_'+str(i)+'.eps', bbox_inches='tight')
    plt.show()
    
    if G_on_projection:
        pos = dict()
        for node in nodes:
            pos[node] = np.array([x_filtered[node], y_filtered[node]])
        
        fig = plt.figure()
        ax = fig.gca()
        nx.draw_networkx_nodes(G.subgraph(graph), pos, node_size=60, node_color=c, cmap=cmap, vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(G.subgraph(graph), pos, width=2.0, edge_color='gray')
        fig.tight_layout()
        #plt.savefig('results/'+name+'/'+name+'_filtered_G_on_projection_'+str(i)+'.pdf', bbox_inches='tight')
        plt.savefig('results/'+name+'/'+name+'_filtered_G_on_projection_'+str(i)+'.eps', bbox_inches='tight')
        plt.show()
