import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

from GRMP import GRMP

from util import getConf

from more_metrics import stress
from sklearn import metrics


#%% Loading data

name = 'digits'

apply_silhouette = True
apply_stress = True

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

conf = getConf(name)


#%% Main code

#gammas = [0.01,0.1,1,10,100]
gammas = np.linspace(1,100,100)

n_gammas = len(gammas)

if apply_stress:
    stresses = np.zeros((n_gammas,))
    
    grmp = GRMP()
    
    for i,gamma in enumerate(gammas):
        grmp.update_gamma(gamma)
        
        X_grmp = grmp.fit_transform(A)
        
        stresses[i] = stress(A, X_grmp)

if apply_silhouette:
    silhouettes = np.zeros((n_gammas,))
    
    grmp = GRMP()
    
    for i,gamma in enumerate(gammas):
        grmp.update_gamma(gamma)
        
        X_grmp = grmp.fit_transform(A)
        
        silhouettes[i] = metrics.silhouette_score(X_grmp, labels)


#%% Plotting results

labelsize = 11

rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize

#if apply_stress:
#    for i,gamma in enumerate(gammas):
#        print(gamma,stresses[i])
#if apply_silhouette:
#    for i,gamma in enumerate(gammas):
#        print(gamma,silhouettes[i])
#if apply_mn:
#    for i,gamma in enumerate(gammas):
#        print(gamma,mns[i])

plt.figure()
if apply_stress:
    plt.plot(gammas, stresses, c='r', label='stress')
if apply_silhouette:
    plt.plot(gammas, silhouettes, c='b', label='silhouette')
plt.legend()
plt.grid('true')
#plt.xlabel('$\gamma$')
plt.savefig('results/'+name+'/'+name+'_gamma.pdf', bbox_inches='tight')
plt.show()
