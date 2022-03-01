import time
from sklearn.metrics import pairwise_distances
import random
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigs,eigsh
from scipy.linalg import block_diag
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np
from scipy.linalg import eigh
#from scipy.sparse import block_diag
from scipy.sparse import bmat
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import adjusted_rand_score
#import cvxopt
from scipy.sparse.linalg import inv



def combined_error_frob_norm(A_CC,A_CU,A_CP,G_C,G_U,G_P):
    """
    Calculate the combined error for the objective function
    :param A_CC: adjacency matrix for CC
    :param A_CU: adjacency matrix for CU
    :param A_CP: adjacency matrix for CP
    :param G_C: cluster indicator matrix for C
    :param G_U: cluster indicator matrix for U
    :param G_P: cluster indicator matrix for P
    :return: objective function error
    """
    G_C = csr_matrix(G_C)
    G_U = csr_matrix(G_U)
    G_P = csr_matrix(G_P)
    GTG_C = G_C.transpose().dot(G_C).toarray()+np.diag([1e-10]*np.sum(G_C.shape[1]))
    GTG_C_inv = inv(csr_matrix(GTG_C))
    S_CC = GTG_C_inv.dot(G_C.transpose().dot(A_CC))
    GTG_U = G_U.transpose().dot(G_U).toarray()+np.diag([1e-10]*np.sum(G_U.shape[1]))
    GTG_U_inv = inv(csr_matrix(GTG_U))
    S_CU = GTG_C_inv.dot(G_C.transpose().dot(A_CU.dot(G_U.dot(GTG_U_inv))))
    # CP
    GTG_P = G_P.transpose().dot(G_P).toarray()+np.diag([1e-10]*np.sum(G_P.shape[1]))
    GTG_P_inv = inv(csr_matrix(GTG_P))
    S_CP = GTG_C_inv.dot(G_C.transpose().dot(A_CP.dot(G_P.dot(GTG_P_inv))))
    S_CC = GTG_C_inv.dot(G_C.transpose().dot(A_CC))
    
    CC_error = (A_CC-G_C.dot(S_CC)).power(2).sum()
    CU_error = (A_CU-G_C.dot(S_CU.dot(G_U.transpose()))).power(2).sum()
    CP_error = (A_CP-G_C.dot(S_CP.dot(G_P.transpose()))).power(2).sum()
    
    A_CC_norm = A_CC.power(2).sum()
    A_CU_norm = A_CU.power(2).sum()
    A_CP_norm = A_CP.power(2).sum()
    
    
    error_rand = CC_error+CU_error+CP_error
    error_scaled = error_rand/(A_CC_norm+A_CU_norm+A_CP_norm)
    return(error_rand)

  
  
  


def NNDSVD_initialisation(A,k):
    """
    Calculate the initialisation for matrix A using NNDSVD
    :param A: adjacency matrix A
    :param k: rank of decomposition
    :return: NNDSVD decomposition initialisation
    """
    U,Sigma,VT = svds(A,k,which='LM')
    if A.shape[0] == A.shape[1]:
        if np.allclose(A.toarray(),A.transpose().toarray()):
            U[:,np.where(U[0,:]<0)[0]]*=-1
            VT[np.where(VT[:,0]<0)[0],:]*=-1
    W = np.zeros((A.shape[0],k))
    H = np.zeros((k,A.shape[1]))
    #W[:,0] = np.sqrt(Sigma[0])*U[:,0]
    #H[0,:] = VT[0,:]*np.sqrt(Sigma[0])
    for j in range(k):
        x = U[:,j]
        y = VT[j,:]
        xp = x.copy()
        xp[x<0] = 0
        xn = x.copy()
        xn[x>0] = 0
        xn = np.abs(xn)
        yp = y.copy()
        yp[y<0] = 0
        yn = y.copy()
        yn[y>0] = 0
        yn = np.abs(yn)
        xpnorm = np.abs(np.sum(xp))
        xnnorm = np.abs(np.sum(xn))
        ypnorm = np.abs(np.sum(yp))
        ynnorm = np.abs(np.sum(yn))
        mp = xpnorm*ypnorm
        mn = xnnorm*ynnorm
        if mp>=mn:
            u = xp/xpnorm
            v = yp/ypnorm
            sigma = mp
        else:
            u = xn/xnnorm
            v = yn/ynnorm
            sigma = mn
        W[:,j] = np.sqrt(Sigma[j]*sigma)* u
        H[j,:] = np.sqrt(Sigma[j]*sigma)* v
    return(W,H)

  
  
  

def cluster_relabel(clust_vec):
    unique_vals,uniq_index = np.unique(clust_vec,return_index=True)
    convert_dic={clust_vec[sorted(uniq_index)[i]]:i for i in range(len(uniq_index))}
    new_clust_vec = [convert_dic[clust_vec[i]] for i in range(len(clust_vec))]
    return new_clust_vec

def A_prob(A,cassign_0,cassign_1):
    cassign_0 = cluster_relabel(cassign_0)
    cassign_1 = cluster_relabel(cassign_1)
    clusters_1 = sorted(set(cassign_1))
    P = np.zeros((len(set(cassign_0)),A.shape[0]))
    P[cassign_0,np.arange(P.shape[1])] = 1.0
    A_csr = A.tocsr()
    A_clusters = np.array([A_csr[:,cassign_1==clustno].max(1).toarray().flatten() for clustno in clusters_1])
    A_clusters = A_clusters.transpose()
    A_clusters = np.dot(P,A_clusters)
    uniq_c0,size_clusters = np.unique(cassign_0,return_counts=True)
    rec_clust_size = 1/size_clusters
    A_clusters = A_clusters*rec_clust_size[:,np.newaxis]
    return A_clusters

def A_prob_new(A,cassign_0,cassign_1):
    cassign_0 = cluster_relabel(cassign_0)
    cassign_1 = cluster_relabel(cassign_1)
    #clusters_1 = sorted(set(cassign_1))
    P_0 = np.zeros((len(set(cassign_0)),A.shape[0]))
    P_1 = np.zeros((A.shape[1],(len(set(cassign_1)))))
    P_0[cassign_0,np.arange(P_0.shape[1])] = 1.0
    P_1[np.arange(P_1.shape[0]),cassign_1] = 1.0
    A_csr = A.tocsr()
    A_clusters = A_csr.dot(P_1)
    A_clusters = np.dot(P_0,A_clusters)
    uniq_c0,size_clusters_0 = np.unique(cassign_0,return_counts=True)
    uniq_c1,size_clusters_1 = np.unique(cassign_1,return_counts=True)
    rec_clust_size_0 = 1/size_clusters_0
    rec_clust_size_1 = 1/size_clusters_1
    A_clusters = A_clusters*rec_clust_size_0[:,np.newaxis]
    A_clusters = A_clusters * rec_clust_size_1
    return A_clusters

from collections import Counter

def top_k(numbers, k=2):
    """The counter.most_common([k]) method works
    in the following way:
    >>> Counter('abracadabra').most_common(3)  
    [('a', 5), ('r', 2), ('b', 2)]
    """

    c = Counter(numbers)
    most_common = [key for key, val in c.most_common(k)]

    return most_common

  
  
  
