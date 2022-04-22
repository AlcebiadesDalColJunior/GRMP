from __future__ import division

import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

warnings.filterwarnings("ignore", message="The MDS API has changed. ``fit`` now constructs an")

class GRMP():
    """Graph Regularization Multidimensional Projection"""

    def __init__(self, neighbors=5, radius=None, gamma=10, initial='phyllotactic'):
        self.neighbors = neighbors
        self.radius = radius
        self.gamma = gamma
        self.initial = initial
        self.similarity_graph = 'knn'
    
    def update_neighbors(self, neighbors):
        self.neighbors = neighbors
    
    def update_radius(self, radius):
        self.radius = radius
    
    def update_gamma(self, gamma):
        self.gamma = gamma
    
    def fit_transform(self, A):
        """Fit the model with A and apply the dimensionality reduction on A."""
        
        if self.radius:
            self.similarity_graph = 'epsilon'
        
        if self.similarity_graph == 'knn':
            W = kneighbors_graph(A, self.neighbors, include_self=False)
        
        if self.similarity_graph == 'epsilon':
            W = radius_neighbors_graph(A, self.radius)
        
        G = nx.from_scipy_sparse_matrix(W)
        
        n_components = nx.number_connected_components(G)
        
        graphs = list(nx.connected_components(G))
        
        control_nodes = []
        control_points = []
        for i,graph in enumerate(graphs):
            nodes = list(graph)
            
            control_node = node_selection(A, nodes)
            control_nodes.append(control_node)
            
            control_point = A[control_node,:]
            control_points.append(control_point)
        
        if n_components > 1:
            mds = MDS(n_components=2, random_state=2)
            
            proj_control_points = mds.fit_transform(np.array(control_points))
        
        n = A.shape[0]
        
        x_init = np.zeros((n,))
        y_init = np.zeros((n,))
        
        x = np.zeros((n,))
        y = np.zeros((n,))
        
        for i,graph in enumerate(graphs):
            n_nodes = len(graph)
            
            nodes = list(graph)
            if n_nodes > 2:
                distances_to_medoid = cdist(A[nodes,:], np.array([control_points[i]]))
                
                if self.initial == 'phyllotactic':
                    pradius = np.max(distances_to_medoid)
                    
                    rho = pradius/np.sqrt(n_nodes)
                    
                    x_init_per_component, y_init_per_component = phyllotactic(n_nodes, rho=rho)
                
                if self.initial == 'grid':
                    pradius = np.max(distances_to_medoid)
                    
                    x_init_per_component, y_init_per_component = grid_distribution(n_nodes, pradius)
                
                nodes = np.array(nodes)
                
                L = nx.laplacian_matrix(G.subgraph(graph))
                
                eigenvector = eigsh(L, k=2, which='SA', tol=1e-04, maxiter=1e8)[1][:,1]
                
                if eigenvector[0] > 0:
                    eigenvector = -eigenvector
                
                graph_ordination = nodes[eigenvector.argsort()]
                
                for j,node in enumerate(graph_ordination):
                    x_init[node] = x_init_per_component[j]
                    y_init[node] = y_init_per_component[j]
                
                gr = graph_regularization(L, self.gamma)
                
                x[nodes] = gr.transform(x_init[nodes])
                y[nodes] = gr.transform(y_init[nodes])
            else:
                x[nodes],y[nodes] = phyllotactic(n_nodes)
                
        self.x_filtered = np.copy(x)
        self.y_filtered = np.copy(y)
        
        if n_components > 1:
            for i,graph in enumerate(graphs):
                nodes = list(graph)
                
                control_node = control_nodes[i]
                
                x[nodes] -= x[control_node]
                y[nodes] -= y[control_node]
                
                x[nodes] += proj_control_points[i,0]
                y[nodes] += proj_control_points[i,1]
        
        X = np.vstack((x,y)).T
        
        self.G = G
        
        self.n_components = n_components
        
        self.graphs = graphs
        
        self.x_init = x_init
        self.y_init = y_init
        
        self.control_nodes = control_nodes
        
        return(X)

def node_selection(A, nodes):
    kmeans = KMeans(n_clusters=1, max_iter=1)
    
    kmeans.fit(A[nodes,:])
    
    centroid = kmeans.cluster_centers_
    
    dist_centroid = np.linalg.norm(A[nodes,:] - centroid, axis=1)
    
    index_component_center = np.argmin(dist_centroid)
    
    node = nodes[index_component_center]
        
    return(node)

def phyllotactic(n_points, xcenter=0, ycenter=0, angle=137.508, rho=0.15):
        """Print a pattern of circles using spiral phyllotactic data"""
        
        phi = angle * (np.pi/180.0)
        
        phy = np.zeros((n_points,2))
        for i in range(n_points):
            r = rho * np.sqrt(i)
            
            theta = i * phi
              
            phy[i,0] = r*np.cos(theta) + xcenter
            phy[i,1] = r*np.sin(theta) + ycenter
        
        focus = np.zeros((n_points,2))
        for i in range(n_points):
            focus[i,0] = 0
            focus[i,1] = -1e8
        
        phy_dist = np.linalg.norm(phy-focus, axis=1)
        
        phy_order = phy_dist.argsort()
        
        x = np.zeros((n_points,))
        y = np.zeros((n_points,))
        
        for i,j in enumerate(phy_order):
            x[i] = phy[j,0]
            y[i] = phy[j,1]
        
        return(x,y)

def grid_distribution(n_points, pradius):
    s = int(np.sqrt(n_points)) + 1
    
    #base = np.linspace(0,pradius,s)
    base = np.linspace(-pradius/2,pradius/2,s)
    
    x, y = np.meshgrid(base, base)
    
    x = x.ravel()
    y = y.ravel()
    
    x = x[:n_points]
    y = y[:n_points]
    
    #x -= np.mean(x)
    #y -= np.mean(y)
    
    return(x,y)
    

class graph_regularization():
    def __init__(self, L, gamma):
        self.M = 20
        
        self.L = L.asfptype()
        self.N = (self.L).shape[0]
        
        self.lmax = eigsh(self.L, k=1, which='LA', return_eigenvectors=False, tol=1e-08, maxiter=1e8)[0]
        self.alpha = self.lmax / 2
        
        self.g = filter_design(gamma)
        
        self.c = cheby_coeff(self.g, self.M, self.alpha)
        
    def transform(self, f):
        pol = cheby_pol(self.L, f, self.M, self.alpha)
        
        f0 = np.zeros((self.N,))
        for i in range(self.N):
            sm = 0
            sm += 0.5 * self.c[0] * f[i]
            
            for k in range(1,self.M):
                sm += self.c[k] * pol[k][i]

            f0[i] = sm
            
        return(f0)
    
def filter_design(gamma):
    filters = lambda x:1/(1+gamma*x)
    
    return(filters)

def cheby_coeff(g, m, alpha):
    N = m + 1
    arange = [0, 2*alpha]
    
    a1 = (arange[1]-arange[0]) / 2
    a2 = (arange[1]+arange[0]) / 2
    
    c = np.zeros((m,))
    for j in range(1,m+1):
        I = np.arange(1,N+1)
        c[j-1] = np.sum(np.multiply(g(a1*np.cos((np.pi*(I-0.5))/N)+a2),np.cos(np.pi*(j-1)*(I-0.5)/N)))*2/N
        
    return(c)

def cheby_pol(L, f, M, alpha):
    pol = []
    pol.append(f)
    pol.append((1/alpha)*L.dot(f)-f)

    for k in range(2,M):
        Lpol = L.dot(pol[k-1])
        pol.append((2/alpha)*Lpol-2*pol[k-1]-pol[k-2])

    return(pol)

if __name__ == "__main__":
    iris = load_iris()
    
    A = iris.data
    
    grmp = GRMP(neighbors=3)
    
    X_grmp = grmp.fit_transform(A)
    
    x_grmp = X_grmp[:,0]
    y_grmp = X_grmp[:,1]
    
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x_grmp, y_grmp)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()