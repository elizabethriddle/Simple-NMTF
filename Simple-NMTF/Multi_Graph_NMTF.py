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
  
  
  
  
  
  
  
  
def simple_multi_NMTF_NNDSVD_wo(A_CC,A_CU,A_CP,A_UU,cluster_sizes,sizes_of_attributes,initial_clustering,verbose=False):
    if not A_CC is None:
        A_CC = csr_matrix(A_CC)
    if not A_CU is None:
        A_CU = csr_matrix(A_CU)
    if not A_CP is None:
        A_CP = csr_matrix(A_CP)
    if not A_UU is None:
        A_UU = csr_matrix(A_UU)
    # Now set the initial clustering and values for G:
    if len(initial_clustering)==0:
        # To make easier, will simply set as random assignment:
        # Initialise using NNDSVD methodology:
        stime = time.time()
        W_CC,H_CC = NNDSVD_initialisation(A_CC.astype("float"),np.max(cluster_sizes))
        W_CU,H_CU = NNDSVD_initialisation(A_CU.astype("float"),np.max(cluster_sizes))
        W_CP,H_CP = NNDSVD_initialisation(A_CP.astype("float"),np.max(cluster_sizes))
        if not A_UU is None:
            W_UU,H_UU = NNDSVD_initialisation(A_UU.astype("float"),np.max(cluster_sizes))
            
        else:
            W_UU = 0
        
        
          
        W_C = (W_CC+W_CU+W_CP)[:,(np.max(cluster_sizes)-cluster_sizes[0]):np.max(cluster_sizes)]/3
        W_U = (H_CU.transpose()+W_UU)[:,(np.max(cluster_sizes)-cluster_sizes[1]):np.max(cluster_sizes)]/2
        W_P = H_CP.transpose()[:,(np.max(cluster_sizes)-cluster_sizes[2]):np.max(cluster_sizes)]
        
        comp_clusters = np.argmax(W_C,1)
        user_clusters = np.argmax(W_U,1)
        port_clusters = np.argmax(W_P,1)
        
        G_C = np.zeros((sizes_of_attributes[0],cluster_sizes[0]))
        G_C[np.arange(G_C.shape[0]),comp_clusters] = 1.0
        G_U = np.zeros((sizes_of_attributes[1],cluster_sizes[1]))
        G_U[np.arange(G_U.shape[0]),user_clusters] = 1.0
        G_P = np.zeros((sizes_of_attributes[2],cluster_sizes[2]))
        G_P[np.arange(G_P.shape[0]),port_clusters] = 1.0
        # Now combine these along diagonal:
        G_C = csr_matrix(G_C)
        G_U = csr_matrix(G_U)
        G_P = csr_matrix(G_P)
        if verbose==True:
            print("Initialise time,",time.time()-stime)
    else:
        G = np.zeros((sum(sizes_of_attributes),sum(cluster_sizes)))
        G[np.arange(G.shape[0]),initial_clustering] = 1
        G_C = G[0:(sizes_of_attributes[0]),0:cluster_sizes[0]]
        G_U = G[sizes_of_attributes[0]:(sizes_of_attributes[0]+sizes_of_attributes[1]),cluster_sizes[0]:(cluster_sizes[0]+cluster_sizes[1])]
        if not A_CP is None:
            G_P = G[(sizes_of_attributes[0]+sizes_of_attributes[1]):(sizes_of_attributes[0]+sizes_of_attributes[1]+sizes_of_attributes[2]),(cluster_sizes[0]+cluster_sizes[1]):(cluster_sizes[0]+cluster_sizes[1]+cluster_sizes[2])]
        G_C = csr_matrix(G_C)
        G_U = csr_matrix(G_U)
        if not A_CP is None:
            G_P = csr_matrix(G_P)
    i=0
    converged=False
    number_repeats = 0
    diff_G_C = 0
    diff_G_U = 0
    if not A_CP is None:
        diff_G_P = 0
    while (converged == False) & (i<100):
        G_C_old = G_C.copy()
        G_U_old = G_U.copy()
        if not A_CP is None:
            G_P_old = G_P.copy()
        diff_G_C_old = diff_G_C
        diff_G_U_old = diff_G_U
        if not A_CP is None:
            diff_G_P_old = diff_G_P
        # First find optimal S values:
        # CC        
        GTG_C = G_C.transpose().dot(G_C).toarray()+np.diag([1e-10]*np.sum(cluster_sizes[0]))
        GTG_C_inv = inv(csr_matrix(GTG_C))
        if not A_CC is None:
            S_CC = GTG_C_inv.dot(G_C.transpose().dot(A_CC))
        # UU
        GTG_U = G_U.transpose().dot(G_U).toarray()+np.diag([1e-10]*np.sum(cluster_sizes[1]))
        GTG_U_inv = inv(csr_matrix(GTG_U))
        if not A_UU is None:
            S_UU = GTG_U_inv.dot(G_U.transpose().dot(A_UU))
        # CU
        if not A_CU is None:
            S_CU = GTG_C_inv.dot(G_C.transpose().dot(A_CU.dot(G_U.dot(GTG_U_inv))))
        # CP
        if not A_CP is None:
            GTG_P = G_P.transpose().dot(G_P).toarray()+np.diag([1e-10]*np.sum(cluster_sizes[2]))
            GTG_P_inv = inv(csr_matrix(GTG_P))
            S_CP = GTG_C_inv.dot(G_C.transpose().dot(A_CP.dot(G_P.dot(GTG_P_inv))))
        
        # Now need to find optimal G!
        # Find pairwise distances:        
        # Find For Computers:
        if not A_CC is None:
            G_dist_CC = pairwise_distances(A_CC,S_CC)
        else:
            G_dist_CC = 0
        if not A_CU is None:
            G_dist_CU_C = pairwise_distances(A_CU,S_CU.dot(G_U.transpose()))
        else:
            G_dist_CU_C = 0
        if not A_CP is None:
            G_dist_CP_C = pairwise_distances(A_CP,S_CP.dot(G_P.transpose()))
        else:
            G_dist_CP_C = 0
        G_C_choice = np.argmin(G_dist_CC+G_dist_CU_C+G_dist_CP_C,axis=1)
        G_C = np.zeros((sizes_of_attributes[0],cluster_sizes[0]))
        G_C[np.arange(G_C.shape[0]),G_C_choice] = 1.0
        G_C = csr_matrix(G_C)
        # Find for Users
        if not A_UU is None:
            G_dist_UU = pairwise_distances(A_UU,S_UU)
        else:
            G_dist_UU = 0
        if not A_CU is None:
            G_dist_CU_U = pairwise_distances(A_CU.transpose(),S_CU.transpose().dot(G_C.transpose()))
        else:
            G_dist_CU_U = 0
        G_U_choice = np.argmin(G_dist_UU+G_dist_CU_U,axis=1)
        G_U = np.zeros((sizes_of_attributes[1],cluster_sizes[1]))
        G_U[np.arange(G_U.shape[0]),G_U_choice] = 1.0
        G_U = csr_matrix(G_U)
        # Find For Ports
        if not A_CP is None:
            G_dist_CP_P = pairwise_distances(A_CP.transpose(),S_CP.transpose().dot(G_C.transpose()))
            G_P_choice = np.argmin(G_dist_CP_P,axis=1)
            G_P = np.zeros((sizes_of_attributes[2],cluster_sizes[2]))
            G_P[np.arange(G_P.shape[0]),G_P_choice] = 1.0
            G_P = csr_matrix(G_P)
        i+=1
        if not A_CP is None:
            if verbose==True:
                print("Iteration,",i,"Diff C",(G_C_old-G_C).power(2).sum(),"Diff U",(G_U_old-G_U).power(2).sum(),"Diff P",(G_P_old-G_P).power(2).sum())
        else:
            if verbose==True:
                print("Iteration,",i,"Diff C",(G_C_old-G_C).power(2).sum(),"Diff U",(G_U_old-G_U).power(2).sum())
        diff_G_C = (G_C_old-G_C).power(2).sum()
        diff_G_U = (G_U_old-G_U).power(2).sum()
        if not A_CP is None:
            diff_G_P = (G_P_old-G_P).power(2).sum()
        
        if (diff_G_C==diff_G_C_old) & (diff_G_U==diff_G_U_old):
            number_repeats += 1
            if number_repeats>4:
                comp_stuck = np.where((G_C_old-G_C).power(2).sum(1)==2)[0]
                user_stuck = np.where((G_U_old-G_U).power(2).sum(1)==2)[0]
                if len(comp_stuck)>0:
                    G_C[comp_stuck,:] = 0
                    G_C[comp_stuck,[random.choice(range(0,cluster_sizes[0])) for i in range(len(comp_stuck))]] = 1
                if len(user_stuck)>0:
                    G_U[user_stuck,:] = 0
                    G_U[user_stuck,[random.choice(range(0,cluster_sizes[1])) for i in range(len(user_stuck))]] = 1
        else:
            number_repeats = 0
            
        if not A_CP is None:
            diff_G = (G_C_old-G_C).power(2).sum()+(G_U_old-G_U).power(2).sum()+(G_P_old-G_P).power(2).sum()
        else:
            diff_G = (G_C_old-G_C).power(2).sum()+(G_U_old-G_U).power(2).sum()
        # Check if converged
        if (diff_G<0.00000001):
            converged = True
    print("Number of iterations for convergence:",i)
    if not A_CP is None:
        return(G_C,G_U,G_P)
    else:
        return(G_C,G_U)
