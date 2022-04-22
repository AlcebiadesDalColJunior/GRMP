from __future__ import division

from sklearn.metrics import euclidean_distances

def stress(A, C):
    dis_A = euclidean_distances(A)
    dis_C = euclidean_distances(C)
    
    #st = ((dis_A.ravel() - dis_C.ravel()) ** 2).sum() / 2
    st = ((dis_A.ravel() - dis_C.ravel()) ** 2).sum() / (dis_A.ravel() ** 2).sum()
    
    return st

def getConf(datasetNumber):
    conf=lambda:None
    
    conf.percentage_of_control_points = 0.05
    
    return(conf)
