# Demonstration of Simple NMTF

Utilising the proposed stochastic block model to generate the network data. As done in our paper, we generate 3 relational data matrices $A_{CC}$, $A_{CU}$ and $A_{CP}$ where the computers (C) have the same clustering accross all 3 matrices. Additionally there is an underlying clustering for the users and ports.


```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import time
from scipy.sparse import csr_matrix, vstack,hstack,lil_matrix, bmat, block_diag,coo_matrix
from sklearn.metrics import pairwise_distances
from scipy.linalg import block_diag as block_diag_l
from sklearn.metrics.cluster import adjusted_rand_score
from importlib import reload
import sys
from scipy.spatial.distance import squareform
import random
from scipy.sparse.linalg import inv

from Simple_NMTF.Multi_Graph_NMTF import simple_multi_NMTF_NNDSVD_wo, simple_multi_NMTF_NNDSVD_newsgroup
from Simple_NMTF.general_functions import combined_error_frob_norm, NNDSVD_initialisation, cluster_relabel, A_prob, A_prob_new, top_k
from Simple_NMTF.node_cluster_similarity.node_cluster_similarity
from Simple_NMTF.sbm import SBM
from Simple_NMTF.sbm_bipartite import SBM_bipartite

```


```python
# Simulate stochastic block models:

comp_number_nodes = 5000
user_number_nodes = 5000
port_number_nodes = 1000
comp_cluster_number = 300
user_cluster_number = 300
port_cluster_number = 100


CC_prob_connection = np.zeros((comp_cluster_number,comp_cluster_number))
CU_prob_connection = np.zeros((comp_cluster_number,user_cluster_number))
CP_prob_connection = np.zeros((comp_cluster_number,port_cluster_number))

comp_break_points=sorted(random.sample(range(1,comp_number_nodes), comp_cluster_number-1))
computer_cluster_sizes = np.array(comp_break_points+[comp_number_nodes])-np.array([0]+comp_break_points)
user_break_points=sorted(random.sample(range(1,user_number_nodes), user_cluster_number-1))
user_cluster_sizes = np.array(user_break_points+[user_number_nodes])-np.array([0]+user_break_points)
port_break_points=sorted(random.sample(range(1,port_number_nodes), port_cluster_number-1))
port_cluster_sizes = np.array(port_break_points+[port_number_nodes])-np.array([0]+port_break_points)

no_connect_CC = np.random.choice(list(range(1,5)),size=comp_cluster_number)
no_connect_CU = np.random.choice(list(range(1,5)),size=comp_cluster_number)
no_connect_CP = np.random.choice(list(range(1,10)),size=comp_cluster_number)

## CC
for i in range(comp_cluster_number):
    which_other_clusters = random.sample(range(comp_cluster_number),no_connect_CC[i])
    prob_of_connection = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9],no_connect_CC[i])
    CC_prob_connection[i,:][which_other_clusters] = prob_of_connection
    CC_prob_connection[:,i][which_other_clusters] = prob_of_connection
## CU
for i in range(comp_cluster_number):
    which_other_clusters = random.sample(range(user_cluster_number),no_connect_CU[i])
    prob_of_connection = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9],no_connect_CU[i])
    CU_prob_connection[i,:][which_other_clusters] = prob_of_connection
# Make sure that all users have probability of connection:
if np.any(CU_prob_connection.sum(0)==0):
    user_issue = np.where(CU_prob_connection.sum(0)==0)[0]
    for i in user_issue:
        which_other_clusters = random.sample(range(comp_cluster_number),1)
        prob_of_connection = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],1)
        CU_prob_connection[:,i][which_other_clusters] = prob_of_connection
## CP
for i in range(comp_cluster_number):
    which_other_clusters = random.sample(range(port_cluster_number),no_connect_CP[i])
    prob_of_connection = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9],no_connect_CP[i])
    CP_prob_connection[i,:][which_other_clusters] = prob_of_connection
# Make sure that all ports have probability of connection:
if np.any(CP_prob_connection.sum(0)==0):
    port_issue = np.where(CP_prob_connection.sum(0)==0)[0]
    for i in port_issue:
        which_other_clusters = random.sample(range(comp_cluster_number),1)
        prob_of_connection = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9],1)
        CP_prob_connection[:,i][which_other_clusters] = prob_of_connection

comp_cluster_assignment = list(np.repeat(np.arange(comp_cluster_number),computer_cluster_sizes))
user_cluster_assignment = list(np.repeat(np.arange(user_cluster_number),user_cluster_sizes))
port_cluster_assignment = list(np.repeat(np.arange(port_cluster_number),port_cluster_sizes))


A_CC_generate = SBM(comp_number_nodes,comp_cluster_number,comp_cluster_assignment,CC_prob_connection).block_matrix
A_CU_generate = SBM_bipartite(comp_number_nodes,user_number_nodes,comp_cluster_assignment,user_cluster_assignment,comp_cluster_assignment,user_cluster_assignment,CU_prob_connection).block_matrix
A_CP_generate = SBM_bipartite(comp_number_nodes,port_number_nodes,comp_cluster_assignment,port_cluster_assignment,comp_cluster_assignment,port_cluster_assignment,CP_prob_connection).block_matrix

# Create CSR matrices:
A_CC_generate_csr = csr_matrix(A_CC_generate)
A_CU_generate_csr = csr_matrix(A_CU_generate)
A_CP_generate_csr = csr_matrix(A_CP_generate)

```


```python
comp_number_nodes = A_CC.shape[0]
user_number_nodes = A_CU.shape[1]
port_number_nodes = A_CP.shape[1]

begin_time = time.time()
G_C_algorithm, G_U_algorithm,G_P_algorithm = Multi_Graph_NMTF.simple_multi_NMTF_NNDSVD_wo(A_CC_generate_csr,A_CU_generate_csr,A_CP_generate_csr,None,[comp_cluster_number,user_cluster_number,port_cluster_number],[comp_number_nodes,user_number_nodes,port_number_nodes],initial_clustering)
alg_time = time.time()-begin_time
print(alg_time)
comp_cluster_algorithm = cluster_relabel(np.where(G_C_algorithm.toarray()==1)[1])
user_cluster_algorithm = cluster_relabel(np.where(G_U_algorithm.toarray()==1)[1])
port_cluster_algorithm = cluster_relabel(np.where(G_P_algorithm.toarray()==1)[1])
# Performance
node_cluster_similarity_CC = node_cluster_similarity(A_CC_generate,comp_cluster_algorithm,comp_cluster_algorithm)/comp_number_nodes
node_cluster_similarity_CU_C = node_cluster_similarity(A_CU_generate,comp_cluster_algorithm,user_cluster_algorithm)/comp_number_nodes
node_cluster_similarity_CU_U = node_cluster_similarity(A_CU_generate.transpose(),user_cluster_algorithm,comp_cluster_algorithm)/user_number_nodes
node_cluster_similarity_CP_C = node_cluster_similarity(A_CP_generate,comp_cluster_algorithm,port_cluster_algorithm)/comp_number_nodes
node_cluster_similarity_CP_P = node_cluster_similarity(A_CP_generate.transpose(),port_cluster_algorithm,comp_cluster_algorithm)/port_number_nodes
node_cluster_similarity_comp = node_cluster_similarity(np.hstack((A_CC_generate,A_CU_generate,A_CP_generate)),comp_cluster_algorithm,np.hstack((comp_cluster_algorithm,user_cluster_algorithm,port_cluster_algorithm)))/comp_number_nodes
node_cluster_similarity_all = ((node_cluster_similarity_comp*comp_number_nodes)+(node_cluster_similarity_CU_U*user_number_nodes)+(node_cluster_similarity_CP_P*port_number_nodes))/(comp_number_nodes+user_number_nodes+port_number_nodes)
## ARI
ARI_comparison = [adjusted_rand_score(comp_cluster_algorithm,comp_cluster_assignment),adjusted_rand_score(user_cluster_algorithm,user_cluster_assignment),adjusted_rand_score(port_cluster_algorithm,port_cluster_assignment),adjusted_rand_score(np.hstack((comp_cluster_algorithm,np.array(user_cluster_algorithm)+comp_cluster_number,np.array(port_cluster_algorithm)+comp_cluster_number+port_cluster_number)),np.hstack((comp_cluster_assignment,np.array(user_cluster_assignment)+comp_cluster_number,np.array(port_cluster_assignment)+comp_cluster_number+port_cluster_number)))]
```


```python

```


```python

```
