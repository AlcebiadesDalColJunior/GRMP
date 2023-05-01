import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GRMP import GRMP

from util import getConf


#%% Loading data

n_frames = 1000

save = True

name = 'digits'

A = np.loadtxt('data/'+name+'.csv')

labels = np.loadtxt('data/'+name+'_labels.csv')

filename = 'results/'+name+'_linear_interpolation.mp4'


#%% Recording the video

conf = getConf(name)

frames = np.linspace(0, 1, n_frames)

grmp = GRMP()

X_grmp = grmp.fit_transform(A)

x_grmp = X_grmp[:,0]
y_grmp = X_grmp[:,1]

x_init = grmp.x_init
y_init = grmp.y_init

X_init = np.vstack((x_init,y_init)).T


#%% Recording the video

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

def func(t, X_init, X_grmp):
    
    X = (1-t) * X_init + t * X_grmp
    
    ax.clear()
    ax.scatter(X[:,0], X[:,1], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(name.capitalize()+' dataset')
    
    return X

fig = plt.figure()
ax = fig.gca()

ani = animation.FuncAnimation(fig, func, frames, fargs=[X_init, X_grmp], repeat=False)

plt.show()

if save:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30)
    
    ani.save(filename, writer=writer, dpi=250)

#t = frames[0]
#
#fig = plt.figure()
#ax = fig.gca()
#X = (1-t) * X_init + t * X_grmp
#ax.clear()
#ax.scatter(X[:,0], X[:,1], c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_ylabel(name.capitalize()+' dataset')