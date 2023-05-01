import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('data/digits.csv')

fig, ax = plt.subplots(4, 5, subplot_kw=dict(xticks=[], yticks=[]))

index = 0
for i in range(4):
    for j in range(5):
        ax[i, j].imshow(A[index].reshape(8, 8), cmap=plt.cm.binary,
                         interpolation='nearest')
    
        index += 1

plt.savefig('results/digits_original.pdf', bbox_inches='tight')
plt.show()
