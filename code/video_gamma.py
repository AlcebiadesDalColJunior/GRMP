import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GRMP import GRMP

from util import getConf

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


#%% Loading data

save = True

name = 'digits'

A = np.loadtxt('datasets/'+name+'.csv')

labels = np.loadtxt('datasets/'+name+'_labels.csv')

filename = 'video/'+name+'_gamma.mp4'


#%% Recording the video

conf = getConf(name)

gammas = np.linspace(0.5,20,1500)

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

def func(gamma):
    
    grmp = GRMP(gamma=gamma)
    
    X_grmp = grmp.fit_transform(A)
    
    ax.clear()
    ax.set_title('GRMP of the '+name.capitalize()+' dataset \quad '+' $\gamma=$ %.2f' % gamma)
    ax.scatter(X_grmp[:,0], X_grmp[:,1], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xlabel(name.capitalize()+' dataset')
    
    return X_grmp

fig = plt.figure()
ax = fig.gca()

ani = animation.FuncAnimation(fig, func, gammas, repeat=False)

if save:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30)
    
    ani.save(filename, writer=writer, dpi=250)

#gamma = gammas[0]
#
#grmp = GRMP(gamma=gamma)
#
#X_grmp = grmp.fit_transform(A)
#
#fig = plt.figure()
#ax = fig.gca()
#ax.set_title('Gamma: %.2f' % gamma)
#ax.scatter(X_grmp[:,0], X_grmp[:,1], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_ylabel(name.capitalize()+' dataset')
#plt.savefig('video/'+name+'_gamma_0.png')
