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
from functools import reduce
import matplotlib.pyplot as plt
import csv

from Simple_NMTF.Multi_Graph_NMTF import simple_multi_NMTF_NNDSVD_wo, simple_multi_NMTF_NNDSVD_newsgroup
from Simple_NMTF.general_functions import combined_error_frob_norm, NNDSVD_initialisation, cluster_relabel, A_prob, A_prob_new, top_k
from Simple_NMTF.node_cluster_similarity.node_cluster_similarity
from Simple_NMTF.sbm import SBM
from Simple_NMTF.sbm_bipartite import SBM_bipartite
from Simple_NMTF.streaming_functions import overlap,overlap_sets,single_node_clustering,streaming_cluster_membership


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
G_C_algorithm, G_U_algorithm,G_P_algorithm = simple_multi_NMTF_NNDSVD_wo(A_CC_generate_csr,A_CU_generate_csr,A_CP_generate_csr,None,[comp_cluster_number,user_cluster_number,port_cluster_number],[comp_number_nodes,user_number_nodes,port_number_nodes],initial_clustering)
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

## Streaming Implementation


```python
from Simple_NMTF.sbm_bipartite_dynamic import SBM_bipartite_dynamic, SBM_dynamic 
```


```python
# Set up for stochastic block generaiton

no_comp = 1000
no_user = 1000
no_port = 500
no_comp_clusters = 20
no_user_clusters = 20
no_port_clusters = 10
window_size = 144
data_length = 1440

comp_number_nodes = no_comp
user_number_nodes = no_user
port_number_nodes = no_port 


number_comp_clusters = no_comp_clusters
number_user_clusters = no_user_clusters
number_port_clusters = no_port_clusters

CC_possible_prob = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.009]
CU_possible_prob = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.009]
CP_possible_prob = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.009]


# Define time step between cluster edge probabilities for each time step
CC_prob_connection_MC = np.zeros((no_comp_clusters,no_comp_clusters))
CU_prob_connection_MC = np.zeros((no_comp_clusters,no_user_clusters))
CP_prob_connection_MC = np.zeros((no_comp_clusters,no_port_clusters))


# Now add probability of connection in the between cluster connection probability:
## Calculate how many entries will have a probability for each cluster
no_connect_CC_MC = np.random.choice(list(range(1,5)),size=no_comp_clusters)
no_connect_CU_MC = np.random.choice(list(range(1,5)),size=no_comp_clusters)
no_connect_CP_MC = np.random.choice(list(range(1,3)),size=no_comp_clusters)


## CC
for i in range(no_comp_clusters):
    which_other_clusters_MC = random.sample(range(no_comp_clusters),no_connect_CC_MC[i])
    prob_of_connection_MC = np.random.choice(CC_possible_prob,no_connect_CC_MC[i])
    CC_prob_connection_MC[i,:][which_other_clusters_MC] = prob_of_connection_MC
    CC_prob_connection_MC[:,i][which_other_clusters_MC] = prob_of_connection_MC
## CU
for i in range(no_comp_clusters):
    which_other_clusters_MC = random.sample(range(no_user_clusters),no_connect_CU_MC[i])
    prob_of_connection_MC = np.random.choice(CU_possible_prob,no_connect_CU_MC[i])
    CU_prob_connection_MC[i,:][which_other_clusters_MC] = prob_of_connection_MC
# Make sure that all users have probability of connection:
if np.any(CU_prob_connection_MC.sum(0)==0):
    user_issue_MC = np.where(CU_prob_connection_MC.sum(0)==0)[0]
    for i in user_issue_MC:
        which_other_clusters_MC = random.sample(range(no_comp_clusters),1)
        prob_of_connection_MC = np.random.choice(CU_possible_prob,1)
        CU_prob_connection_MC[:,i][which_other_clusters_MC] = prob_of_connection_MC
## CP
for i in range(no_comp_clusters):
    which_other_clusters_MC = random.sample(range(no_port_clusters),no_connect_CP_MC[i])
    prob_of_connection_MC = np.random.choice(CP_possible_prob,no_connect_CP_MC[i])
    CP_prob_connection_MC[i,:][which_other_clusters_MC] = prob_of_connection_MC
# Make sure that all ports have probability of connection:
if np.any(CP_prob_connection_MC.sum(0)==0):
    port_issue_MC = np.where(CP_prob_connection_MC.sum(0)==0)[0]
    for i in port_issue_MC:
        which_other_clusters_MC = random.sample(range(no_comp_clusters),1)
        prob_of_connection_MC = np.random.choice(CP_possible_prob,1)
        CP_prob_connection_MC[:,i][which_other_clusters_MC] = prob_of_connection_MC
        
```


```python
# Define cluster adaptation
comp_community_adaptation = np.array(streaming_cluster_membership(data_length,[2]*no_comp,no_comp_clusters)).transpose()
user_community_adaptation = np.array(streaming_cluster_membership(data_length,[2]*no_user,no_user_clusters)).transpose()
port_community_adaptation = np.array(streaming_cluster_membership(data_length,[2]*no_port,no_port_clusters)).transpose()
```


```python
# Times of Change
many_exp_times_of_change_comp = [sorted([np.min(np.where(comp_community_adaptation[:,i]==x)) for x in set(comp_community_adaptation[:,i].tolist())]) for i in range(no_comp)]
many_exp_times_of_change_user = [sorted([np.min(np.where(user_community_adaptation[:,i]==x)) for x in set(user_community_adaptation[:,i].tolist())]) for i in range(no_user)]
many_exp_times_of_change_port = [sorted([np.min(np.where(port_community_adaptation[:,i]==x)) for x in set(port_community_adaptation[:,i].tolist())]) for i in range(no_port)]
```


```python
# Set up document to write to as adapt

update_data_file_performance = "".join(["DynamicSimClustering/Clustering_Performance_ManyChange.txt"])
with open(update_data_file_performance, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(["Bin","Time","Overall Sim Previous","ARI Comp","ARI User","ARI Port","NMI Comp","NMI User","NMI Port","ARI Comp True","ARI User True","ARI Port True","NMI Comp True","NMI User True","NMI Port True","NCS CC","NCS CU C","NCS CU U","NCS CP C","NCS CP P","NCS comp","NCS all","No Comp Cluster Change","No User Cluster Change","No Port Cluster Change"])

update_data_file_performance_restart = "".join(["DynamicSimClustering/Clustering_Performance_Restart_ManyChange.txt"])
with open(update_data_file_performance_restart, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(["Bin","Time","Overall Sim Previous","ARI Comp","ARI User","ARI Port","NMI Comp","NMI User","NMI Port","ARI Comp True","ARI User True","ARI Port True","NMI Comp True","NMI User True","NMI Port True","NCS CC","NCS CU C","NCS CU U","NCS CP C","NCS CP P","NCS comp","NCS all","No Comp Cluster Change","No User Cluster Change","No Port Cluster Change"])

```

#### Run Clustering


```python
# Similate for each time point
edges_CC = {}
edges_CU = {}
edges_CP = {}
vertices_CC_comp = set()
vertices_CU_comp = set()
vertices_CP_comp = set()
vertices_CU_user = set()
vertices_CP_port = set()
vertices_inactive_CC_comp = set()
vertices_inactive_CU_comp = set()
vertices_inactive_CP_comp = set()
vertices_inactive_CU_user = set()
vertices_inactive_CP_port = set()

common_comp = list(range(no_comp))
common_user = list(range(no_user))
common_port = list(range(no_port))
cluster_performance_collection = []
for binno in range(data_length):
    A_CC_generate_MC_time_step = SBM_dynamic(no_comp,no_comp_clusters,comp_community_adaptation[binno,:].tolist(),CC_prob_connection_MC).block_matrix
    A_CU_generate_MC_time_step = SBM_bipartite_dynamic(no_comp,no_user,no_comp_clusters,no_user_clusters,comp_community_adaptation[binno,:].tolist(),user_community_adaptation[binno,:].tolist(),CU_prob_connection_MC).block_matrix
    A_CP_generate_MC_time_step = SBM_bipartite_dynamic(no_comp,no_port,no_comp_clusters,no_comp_clusters,comp_community_adaptation[binno,:].tolist(),port_community_adaptation[binno,:].tolist(),CP_prob_connection_MC).block_matrix
    
    
    CC_new_edges = np.where(A_CC_generate_MC_time_step==1)
    CU_new_edges = np.where(A_CU_generate_MC_time_step==1)
    CP_new_edges = np.where(A_CP_generate_MC_time_step==1)

    ttime = time.time()
    if binno in range(window_size):
        
        ## Update CC
        vertices_CC_comp.update(CC_new_edges[0].tolist())
        for i in range(len(CC_new_edges[0])):
            edges_CC[tuple(sorted([CC_new_edges[0][i],CC_new_edges[1][i]]))] = binno
        ## Update CU
        vertices_CU_comp.update(CU_new_edges[0].tolist())
        vertices_CU_user.update(CU_new_edges[1].tolist())
        for i in range(len(CU_new_edges[0])):
            edges_CU[tuple([CU_new_edges[0][i],CU_new_edges[1][i]])] = binno
        ## Update CP
        vertices_CP_comp.update(CP_new_edges[0].tolist())
        vertices_CP_port.update(CP_new_edges[1].tolist())
        for i in range(len(CP_new_edges[0])):
            edges_CP[tuple([CP_new_edges[0][i],CP_new_edges[1][i]])] = binno
        if binno==(window_size-1):
            #Initial Clustering

            edges_CC_common = np.array([[x[0],x[1]] for x in edges_CC.keys()])
            edges_CU_common = np.array([[x[0],x[1]] for x in edges_CU.keys()])
            edges_CP_common = np.array([[x[0],x[1]] for x in edges_CP.keys()])

            A_CC = csr_matrix(([1]*(len(edges_CC_common)*2),(edges_CC_common.flatten('F'),edges_CC_common[:,[1,0]].flatten('F'))),shape=(no_comp,no_comp))
            A_CC[A_CC>1] = 1
            A_CC = A_CC.tolil()
            A_CU = csr_matrix(([1]*(len(edges_CU_common)),(edges_CU_common[:,0],edges_CU_common[:,1])),shape=(no_comp,no_user))
            A_CU = A_CU.tolil()
            A_CU[A_CU>1] = 1
            A_CP = csr_matrix(([1]*(len(edges_CP_common)),(edges_CP_common[:,0],edges_CP_common[:,1])),shape=(no_comp,no_port))
            A_CP = A_CP.tolil()
            A_CP[A_CP>1] = 1

            # Perform Simple Clustering with NNDSVD initialisation
            begin_time = time.time()
            G_C_algorithm, G_U_algorithm,G_P_algorithm = Multi_Graph_NMTF.simple_multi_NMTF_NNDSVD_wo(A_CC,A_CU,A_CP,None,[number_comp_clusters,number_user_clusters,number_port_clusters],[len(common_comp),len(common_user),len(common_port)],[])
            alg_time = time.time()-begin_time
            comp_cluster_list = cluster_relabel(np.where(G_C_algorithm.toarray()==1)[1])
            user_cluster_list = cluster_relabel(np.where(G_U_algorithm.toarray()==1)[1])
            port_cluster_list = cluster_relabel(np.where(G_P_algorithm.toarray()==1)[1])
            
            
            clustering_overall = np.hstack((comp_cluster_list,np.array(user_cluster_list)+number_comp_clusters,np.array(port_cluster_list)+(number_comp_clusters+number_port_clusters)))
            #similarity_previous = adjusted_rand_score(clustering_overall,previous_clustering)

            clustering_comp = {common_comp[x]:comp_cluster_list[x] for x in range(len(common_comp))}
            clustering_user = {common_user[x]:user_cluster_list[x] for x in range(len(common_user))}
            clustering_port = {common_port[x]:port_cluster_list[x] for x in range(len(common_port))}

            
            comp_cluster_list_restart = comp_cluster_list.copy()
            user_cluster_list_restart = user_cluster_list.copy()
            port_cluster_list_restart = port_cluster_list.copy()
            
            
            comp_cluster_index = np.array(range(len(set(comp_cluster_list))))
            comp_column_index = {x:x for x in range(len(comp_cluster_index))}
            user_cluster_index = np.array(range(len(set(user_cluster_list))))
            user_column_index = {x:x for x in range(len(user_cluster_index))}
            port_cluster_index = np.array(range(len(set(port_cluster_list))))
            port_column_index = {x:x for x in range(len(port_cluster_index))}


            clustering_comp = {common_comp[x]:comp_cluster_list[x] for x in range(len(common_comp))}
            clustering_user = {common_user[x]:user_cluster_list[x] for x in range(len(common_user))}
            clustering_port = {common_port[x]:port_cluster_list[x] for x in range(len(common_port))}

            index_age_comp = {x:1 for x in range(len(set(comp_cluster_list)))}
            index_age_user = {x:1 for x in range(len(set(user_cluster_list)))}
            index_age_port = {x:1 for x in range(len(set(port_cluster_list)))}

            cluster_dictionary_new_comp=cluster_dictionary_creation(comp_cluster_list,common_comp)
            cluster_dictionary_new_user=cluster_dictionary_creation(user_cluster_list,common_user)
            cluster_dictionary_new_port=cluster_dictionary_creation(port_cluster_list,common_port)

            cluster_dictionary_new_comp_restart=cluster_dictionary_creation(comp_cluster_list,common_comp)
            cluster_dictionary_new_user_restart=cluster_dictionary_creation(user_cluster_list,common_user)
            cluster_dictionary_new_port_restart=cluster_dictionary_creation(port_cluster_list,common_port)

            non_zero_clusters = [np.sum(G_C_algorithm.sum(0)!=0),np.sum(G_U_algorithm.sum(0)!=0),np.sum(G_P_algorithm.sum(0)!=0)]
            single_entity_clusters = [np.sum(G_C_algorithm.sum(0)==1),np.sum(G_U_algorithm.sum(0)==1),np.sum(G_P_algorithm.sum(0)==1)]
            
            
            
            
            
            common_comp_old = set(common_comp)
            common_user_old = set(common_user)
            common_port_old = set(common_user)
                
            print("Number of Non Zero Clusters:",non_zero_clusters)
            print("Number of Single Enity Clusters:",single_entity_clusters)

            print("Computer Cluster Sizes:",G_C_algorithm.sum(0)[G_C_algorithm.sum(0)>0].astype("int"))
            print("User Cluster Sizes:",G_U_algorithm.sum(0)[G_U_algorithm.sum(0)>0].astype("int"))
            print("Port Cluster Sizes:",G_P_algorithm.sum(0)[G_P_algorithm.sum(0)>0].astype("int"))

            print("Bin",binno,"Time:",time.time()-ttime)
    else:
        # Collect new data
        ## Update CC
        vertices_CC_comp.update(CC_new_edges[0].tolist())
        for i in range(len(CC_new_edges[0])):
            edges_CC[tuple(sorted([CC_new_edges[0][i],CC_new_edges[1][i]]))] = binno
        ## Update CU
        vertices_CU_comp.update(CU_new_edges[0].tolist())
        vertices_CU_user.update(CU_new_edges[1].tolist())
        for i in range(len(CU_new_edges[0])):
            edges_CU[tuple([CU_new_edges[0][i],CU_new_edges[1][i]])] = binno
        ## Update CP
        vertices_CP_comp.update(CP_new_edges[0].tolist())
        vertices_CP_port.update(CP_new_edges[1].tolist())
        for i in range(len(CP_new_edges[0])):
            edges_CP[tuple([CP_new_edges[0][i],CP_new_edges[1][i]])] = binno
            
            
        bin_to_remove = binno-window_size+1
        # Now Remove Old Edges from edgelist:
        edges_removed_CC = {tuple(k) for k,v in edges_CC.items() if v==bin_to_remove}
        edges_removed_CP = {tuple(k) for k,v in edges_CP.items() if v==bin_to_remove}
        edges_removed_CU = {tuple(k) for k,v in edges_CU.items() if v==bin_to_remove}

        edges_CC = {k:v for k,v in edges_CC.items() if v!=bin_to_remove}
        edges_CP = {k:v for k,v in edges_CP.items() if v!=bin_to_remove}
        edges_CU = {k:v for k,v in edges_CU.items() if v!=bin_to_remove}
        
        # Remove inactive nodes from vertice list and add to inactive nodes
        vertices_CC_comp_new = {y for x in edges_CC for y in x}
        vertices_CU_comp_new = {x[0] for x in edges_CU}
        vertices_CP_comp_new = {x[0] for x in edges_CP}
        vertices_CP_port_new = {x[1] for x in edges_CP}
        vertices_CU_user_new = {x[1] for x in edges_CU}

        # Find the new inactive nodes
        vertices_inactive_CC_comp_new = vertices_CC_comp - vertices_CC_comp_new
        vertices_inactive_CU_comp_new = vertices_CU_comp - vertices_CU_comp_new
        vertices_inactive_CP_comp_new = vertices_CP_comp - vertices_CP_comp_new
        vertices_inactive_CP_port_new = vertices_CP_port - vertices_CP_port_new
        vertices_inactive_CU_user_new = vertices_CU_user - vertices_CU_user_new

        # Rename vertices:
        vertices_CC_comp = vertices_CC_comp_new
        vertices_CU_comp = vertices_CU_comp_new
        vertices_CP_comp = vertices_CP_comp_new
        vertices_CP_port = vertices_CP_port_new
        vertices_CU_user = vertices_CU_user_new

        # Add these to the inactive nodes
        vertices_inactive_CC_comp = vertices_inactive_CC_comp | vertices_inactive_CC_comp_new
        vertices_inactive_CU_comp = vertices_inactive_CC_comp | vertices_inactive_CC_comp_new
        vertices_inactive_CP_comp = vertices_inactive_CP_comp | vertices_inactive_CP_comp_new
        vertices_inactive_CU_user = vertices_inactive_CU_user | vertices_inactive_CU_user_new
        vertices_inactive_CP_port = vertices_inactive_CP_port | vertices_inactive_CP_port_new
        
        ## SETUP for Clustering
        edges_CC_common = np.array([[x[0],x[1]] for x in edges_CC.keys()])
        edges_CU_common = np.array([[x[0],x[1]] for x in edges_CU.keys()])
        edges_CP_common = np.array([[x[0],x[1]] for x in edges_CP.keys()])

        A_CC = csr_matrix(([1]*(len(edges_CC_common)*2),(edges_CC_common.flatten('F'),edges_CC_common[:,[1,0]].flatten('F'))),shape=(no_comp,no_comp))
        A_CC[A_CC>1] = 1
        A_CC = A_CC.tolil()
        A_CU = csr_matrix(([1]*(len(edges_CU_common)),(edges_CU_common[:,0],edges_CU_common[:,1])),shape=(no_comp,no_user))
        A_CU = A_CU.tolil()
        A_CU[A_CU>1] = 1
        A_CP = csr_matrix(([1]*(len(edges_CP_common)),(edges_CP_common[:,0],edges_CP_common[:,1])),shape=(no_comp,no_port))
        A_CP = A_CP.tolil()
        A_CP[A_CP>1] = 1

        # Now create vector with previous clustering:
        previous_comp_clustering = comp_cluster_list.copy()
        previous_user_clustering = user_cluster_list.copy()
        previous_port_clustering = port_cluster_list.copy()
        
        # Comp
        clustering_comp_old = clustering_comp.copy()
        clustering_user_old = clustering_user.copy()
        clustering_port_old = clustering_port.copy()

        previous_clustering = []
        previous_clustering.extend((previous_comp_clustering))
        previous_clustering.extend((np.array(previous_user_clustering)+number_comp_clusters).tolist())
        previous_clustering.extend((np.array(previous_port_clustering)+(number_comp_clusters+number_user_clusters)).tolist())

        # Begin Clustering
        comp_cluster_list_old = comp_cluster_list.copy()
        user_cluster_list_old = user_cluster_list.copy()
        port_cluster_list_old = port_cluster_list.copy()
        # Perform Simple Clustering using previous clustering
        begin_time = time.time()
        G_C_algorithm, G_U_algorithm,G_P_algorithm = simple_multi_NMTF_NNDSVD_wo(A_CC,A_CU,A_CP,None,[number_comp_clusters,number_user_clusters,number_port_clusters],[len(common_comp),len(common_user),len(common_port)],previous_clustering)
        alg_time = time.time()-begin_time
        comp_cluster_list = cluster_relabel(np.where(G_C_algorithm.toarray()==1)[1])
        user_cluster_list = cluster_relabel(np.where(G_U_algorithm.toarray()==1)[1])
        port_cluster_list = cluster_relabel(np.where(G_P_algorithm.toarray()==1)[1])

        clustering_overall = np.hstack((comp_cluster_list,np.array(user_cluster_list)+number_comp_clusters,np.array(port_cluster_list)+(number_comp_clusters+number_port_clusters)))
        similarity_previous = adjusted_rand_score(clustering_overall,previous_clustering)

        clustering_comp = {common_comp[x]:comp_cluster_list[x] for x in range(len(common_comp))}
        clustering_user = {common_user[x]:user_cluster_list[x] for x in range(len(common_user))}
        clustering_port = {common_port[x]:port_cluster_list[x] for x in range(len(common_port))}

        
        
        # Performance of Clustering:
        NCS_CC_C = node_cluster_similarity(A_CC,comp_cluster_list,comp_cluster_list)/comp_number_nodes
        NCS_CU_C = node_cluster_similarity(A_CU,comp_cluster_list,user_cluster_list)/comp_number_nodes
        NCS_CU_U = node_cluster_similarity(A_CU.transpose(),user_cluster_list,comp_cluster_list)/user_number_nodes
        NCS_CP_C = node_cluster_similarity(A_CP,comp_cluster_list,port_cluster_list)/comp_number_nodes
        NCS_CP_P = node_cluster_similarity(A_CP.transpose(),port_cluster_list,comp_cluster_list)/port_number_nodes
        NCS_comp = node_cluster_similarity(np.hstack((A_CC.toarray(), A_CU.toarray(),A_CP.toarray())),comp_cluster_list,np.hstack((comp_cluster_list,user_cluster_list,port_cluster_list)))/comp_number_nodes
        NCS_all = ((NCS_comp*comp_number_nodes)+(NCS_CU_U*user_number_nodes)+(NCS_CP_P*port_number_nodes))/(comp_number_nodes+user_number_nodes+port_number_nodes)


        # Similarity Individual to Previous:
        ARI_comp = adjusted_rand_score(comp_cluster_list,previous_comp_clustering)
        ARI_user = adjusted_rand_score(user_cluster_list,previous_user_clustering)
        ARI_port = adjusted_rand_score(port_cluster_list,previous_port_clustering)
        
        NMI_comp = normalized_mutual_info_score(comp_cluster_list,previous_comp_clustering)
        NMI_user = normalized_mutual_info_score(user_cluster_list,previous_user_clustering)
        NMI_port = normalized_mutual_info_score(port_cluster_list,previous_port_clustering)
        
        # Similarity To True
        ARI_comp_true = adjusted_rand_score(comp_cluster_list,comp_community_adaptation[binno,:].tolist())
        ARI_user_true = adjusted_rand_score(user_cluster_list,user_community_adaptation[binno,:].tolist())
        ARI_port_true = adjusted_rand_score(port_cluster_list,port_community_adaptation[binno,:].tolist())
        
        NMI_comp_true = normalized_mutual_info_score(comp_cluster_list,comp_community_adaptation[binno,:].tolist())
        NMI_user_true = normalized_mutual_info_score(user_cluster_list,user_community_adaptation[binno,:].tolist())
        NMI_port_true = normalized_mutual_info_score(port_cluster_list,port_community_adaptation[binno,:].tolist())
        
        cluster_performance_collection.append([binno,similarity_previous,ARI_comp,ARI_user,ARI_port,NMI_comp,NMI_user,NMI_port,ARI_comp_true,ARI_user_true,ARI_port_true,NMI_comp_true,NMI_user_true,NMI_port_true,NCS_CC_C,NCS_CU_C,NCS_CU_U,NCS_CP_C,NCS_CP_P,NCS_comp,NCS_all])
        
        # Computer Overlap
        cluster_dictionary_old_comp = cluster_dictionary_new_comp.copy()
        cluster_dictionary_new_comp = cluster_dictionary_creation(comp_cluster_list,common_comp)
        overlap_matrix_comp = overlap_sets(cluster_dictionary_old_comp,cluster_dictionary_new_comp,set(common_comp_old),set(common_comp))
        comp_overlap_old = {x: np.where(np.array(overlap_matrix_comp).transpose()[x,:]>0.5)[0].tolist() for x in set(comp_cluster_list)}
        comp_same_cluster = [comp_cluster_list[x] in comp_overlap_old[comp_cluster_list[x]] for x in range(no_comp)]
        number_comp_cluster_change = no_comp-sum(comp_same_cluster)
        
        cluster_dictionary_old_user = cluster_dictionary_new_user.copy()
        cluster_dictionary_new_user = cluster_dictionary_creation(user_cluster_list,common_user)
        overlap_matrix_user = overlap_sets(cluster_dictionary_old_user,cluster_dictionary_new_user,set(common_user_old),set(common_user))
        user_overlap_old = {x: np.where(np.array(overlap_matrix_user).transpose()[x,:]>0.5)[0].tolist() for x in set(user_cluster_list)}
        user_same_cluster = [user_cluster_list[x] in user_overlap_old[user_cluster_list[x]] for x in range(no_user)]
        number_user_cluster_change = no_user-sum(user_same_cluster)
        
        cluster_dictionary_old_port = cluster_dictionary_new_port.copy()
        cluster_dictionary_new_port = cluster_dictionary_creation(port_cluster_list,common_port)
        overlap_matrix_port = overlap_sets(cluster_dictionary_old_port,cluster_dictionary_new_port,set(common_port_old),set(common_port))
        port_overlap_old = {x: np.where(np.array(overlap_matrix_port).transpose()[x,:]>0.5)[0].tolist() for x in set(port_cluster_list)}
        port_same_cluster = [port_cluster_list[x] in port_overlap_old[port_cluster_list[x]] for x in range(no_port)]
        number_port_cluster_change = no_port-sum(port_same_cluster )  


            
        
        # Document Cluster Performance:
        with open(update_data_file_performance, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([binno,alg_time,similarity_previous,ARI_comp,ARI_user,ARI_port,NMI_comp,NMI_user,NMI_port,ARI_comp_true,ARI_user_true,ARI_port_true,NMI_comp_true,NMI_user_true,NMI_port_true,NCS_CC_C,NCS_CU_C,NCS_CU_U,NCS_CP_C,NCS_CP_P,NCS_comp,NCS_all,number_comp_cluster_change,number_user_cluster_change,number_port_cluster_change])
        
        
        
        
        ##########
        # Restart Method
        begin_time = time.time()
        G_C_algorithm_restart, G_U_algorithm_restart,G_P_algorithm_restart = Multi_Graph_NMTF.simple_multi_NMTF_NNDSVD_wo(A_CC,A_CU,A_CP,None,[number_comp_clusters,number_user_clusters,number_port_clusters],[len(common_comp),len(common_user),len(common_port)],[])
        alg_time = time.time()-begin_time
        previous_comp_clustering_restart = comp_cluster_list_restart.copy()
        previous_user_clustering_restart = user_cluster_list_restart.copy()
        previous_port_clustering_restart = port_cluster_list_restart.copy()
        previous_clustering_restart = []
        previous_clustering_restart.extend((previous_comp_clustering_restart))
        previous_clustering_restart.extend((np.array(previous_user_clustering_restart)+number_comp_clusters).tolist())
        previous_clustering_restart.extend((np.array(previous_port_clustering_restart)+(number_comp_clusters+number_user_clusters)).tolist())
        comp_cluster_list_restart = cluster_relabel(np.where(G_C_algorithm_restart.toarray()==1)[1])
        user_cluster_list_restart = cluster_relabel(np.where(G_U_algorithm_restart.toarray()==1)[1])
        port_cluster_list_restart = cluster_relabel(np.where(G_P_algorithm_restart.toarray()==1)[1])
        clustering_overall_restart = np.hstack((comp_cluster_list_restart,np.array(user_cluster_list_restart)+number_comp_clusters,np.array(port_cluster_list)+(number_comp_clusters+number_port_clusters)))
        similarity_previous_restart = adjusted_rand_score(clustering_overall_restart,previous_clustering_restart)
        # Performance of Clustering:
        NCS_CC_C = node_cluster_similarity(A_CC,comp_cluster_list_restart,comp_cluster_list_restart)/comp_number_nodes
        NCS_CU_C = node_cluster_similarity(A_CU,comp_cluster_list_restart,user_cluster_list_restart)/comp_number_nodes
        NCS_CU_U = node_cluster_similarity(A_CU.transpose(),user_cluster_list_restart,comp_cluster_list_restart)/user_number_nodes
        NCS_CP_C = node_cluster_similarity(A_CP,comp_cluster_list_restart,port_cluster_list_restart)/comp_number_nodes
        NCS_CP_P = node_cluster_similarity(A_CP.transpose(),port_cluster_list_restart,comp_cluster_list_restart)/port_number_nodes
        NCS_comp = node_cluster_similarity(np.hstack((A_CC.toarray(), A_CU.toarray(),A_CP.toarray())),comp_cluster_list_restart,np.hstack((comp_cluster_list_restart,user_cluster_list_restart,port_cluster_list_restart)))/comp_number_nodes
        NCS_all = ((NCS_comp*comp_number_nodes)+(NCS_CU_U*user_number_nodes)+(NCS_CP_P*port_number_nodes))/(comp_number_nodes+user_number_nodes+port_number_nodes)
        # Similarity Individual to Previous:
        ARI_comp = adjusted_rand_score(comp_cluster_list_restart,previous_comp_clustering_restart)
        ARI_user = adjusted_rand_score(user_cluster_list_restart,previous_user_clustering_restart)
        ARI_port = adjusted_rand_score(port_cluster_list_restart,previous_port_clustering_restart)
        #NMI
        NMI_comp = normalized_mutual_info_score(comp_cluster_list_restart,previous_comp_clustering_restart)
        NMI_user = normalized_mutual_info_score(user_cluster_list_restart,previous_user_clustering_restart)
        NMI_port = normalized_mutual_info_score(port_cluster_list_restart,previous_port_clustering_restart)
        # Similarity To True
        ARI_comp_true = adjusted_rand_score(comp_cluster_list_restart,comp_community_adaptation[binno,:].tolist())
        ARI_user_true = adjusted_rand_score(user_cluster_list_restart,user_community_adaptation[binno,:].tolist())
        ARI_port_true = adjusted_rand_score(port_cluster_list_restart,port_community_adaptation[binno,:].tolist())
        #NMI
        NMI_comp_true = normalized_mutual_info_score(comp_cluster_list_restart,comp_community_adaptation[binno,:].tolist())
        NMI_user_true = normalized_mutual_info_score(user_cluster_list_restart,user_community_adaptation[binno,:].tolist())
        NMI_port_true = normalized_mutual_info_score(port_cluster_list_restart,port_community_adaptation[binno,:].tolist())
        
        # Computer Overlap
        cluster_dictionary_old_comp_restart = cluster_dictionary_new_comp_restart.copy()
        cluster_dictionary_new_comp_restart = cluster_dictionary_creation(comp_cluster_list_restart,common_comp)
        overlap_matrix_comp_restart = overlap_sets(cluster_dictionary_old_comp_restart,cluster_dictionary_new_comp_restart,set(common_comp_old),set(common_comp))
        comp_overlap_old_restart = {x: np.where(np.array(overlap_matrix_comp_restart).transpose()[x,:]>0.5)[0].tolist() for x in set(comp_cluster_list_restart)}
        comp_same_cluster_restart = [comp_cluster_list_restart[x] in comp_overlap_old_restart[comp_cluster_list_restart[x]] for x in range(no_comp)]
        number_comp_cluster_change = no_comp-sum(comp_same_cluster_restart)
        
        cluster_dictionary_old_user_restart = cluster_dictionary_new_user.copy()
        cluster_dictionary_new_user_restart = cluster_dictionary_creation(user_cluster_list_restart,common_user)
        overlap_matrix_user_restart = overlap_sets(cluster_dictionary_old_user_restart,cluster_dictionary_new_user_restart,set(common_user_old),set(common_user))
        user_overlap_old_restart = {x: np.where(np.array(overlap_matrix_user_restart).transpose()[x,:]>0.5)[0].tolist() for x in set(user_cluster_list_restart)}
        user_same_cluster_restart = [user_cluster_list_restart[x] in user_overlap_old_restart[user_cluster_list_restart[x]] for x in range(no_user)]
        number_user_cluster_change = no_user-sum(user_same_cluster_restart)
        
        cluster_dictionary_old_port_restart = cluster_dictionary_new_port_restart.copy()
        cluster_dictionary_new_port_restart = cluster_dictionary_creation(port_cluster_list_restart,common_port)
        overlap_matrix_port_restart = overlap_sets(cluster_dictionary_old_port_restart,cluster_dictionary_new_port_restart,set(common_port_old),set(common_port))
        port_overlap_old_restart = {x: np.where(np.array(overlap_matrix_port_restart).transpose()[x,:]>0.5)[0].tolist() for x in set(port_cluster_list_restart)}
        port_same_cluster_restart = [port_cluster_list_restart[x] in port_overlap_old_restart[port_cluster_list_restart[x]] for x in range(no_port)]
        number_port_cluster_change = no_port-sum(port_same_cluster_restart)   
        # Document Cluster Performance:
        with open(update_data_file_performance_restart, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([binno,alg_time,similarity_previous,ARI_comp,ARI_user,ARI_port,NMI_comp,NMI_user,NMI_port,ARI_comp_true,ARI_user_true,ARI_port_true,NMI_comp_true,NMI_user_true,NMI_port_true,NCS_CC_C,NCS_CU_C,NCS_CU_U,NCS_CP_C,NCS_CP_P,NCS_comp,NCS_all,number_comp_cluster_change,number_user_cluster_change,number_port_cluster_change])

                
        non_zero_clusters = [np.sum(G_C_algorithm.sum(0)!=0),np.sum(G_U_algorithm.sum(0)!=0),np.sum(G_P_algorithm.sum(0)!=0)]
        single_entity_clusters = [np.sum(G_C_algorithm.sum(0)==1),np.sum(G_U_algorithm.sum(0)==1),np.sum(G_P_algorithm.sum(0)==1)]
            
            
                
        print("Number of Non Zero Clusters:",non_zero_clusters)
        print("Number of Single Enity Clusters:",single_entity_clusters)

        print("Computer Cluster Sizes:",G_C_algorithm.sum(0)[G_C_algorithm.sum(0)>0].astype("int"))
        print("User Cluster Sizes:",G_U_algorithm.sum(0)[G_U_algorithm.sum(0)>0].astype("int"))
        print("Port Cluster Sizes:",G_P_algorithm.sum(0)[G_P_algorithm.sum(0)>0].astype("int"))

        print("Bin",binno,"Time:",time.time()-ttime)



```


```python

```

### Display Results


```python
cluster_performance_collection_array = np.array(cluster_performance_collection)
```

#### Similarity to Previous ARI


```python
plt.figure(figsize=(15, 3))

plt.subplot(141)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,1].flatten())
plt.title("All")
plt.subplot(142)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,2].flatten())
plt.title("C")
plt.subplot(143)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,3].flatten())
plt.title("U")
plt.subplot(144)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,4].flatten())
plt.title("P")
plt.suptitle('Sim Prev')
plt.show()
```

#### Similarity to True ARI


```python

plt.figure(figsize=(15, 3))


plt.subplot(131)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,8].flatten())
plt.title("C")
plt.subplot(132)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,9].flatten())
plt.title("U")
plt.subplot(133)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,10].flatten())
plt.title("P")
plt.suptitle('Sim True ARI')
plt.show()
```

#### Similarity to Previous NMI


```python

plt.figure(figsize=(21, 3))


plt.subplot(171)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,14].flatten())
plt.title("CC")
plt.subplot(172)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,15].flatten())
plt.title("CU C")
plt.subplot(173)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,16].flatten())
plt.title("CU U")
plt.subplot(174)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,17].flatten())
plt.title("CP C")
plt.subplot(175)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,18].flatten())
plt.title("CP P")
plt.subplot(176)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,19].flatten())
plt.title("Comp")
plt.subplot(177)
plt.plot(cluster_performance_collection_array[:,0].flatten(),cluster_performance_collection_array[:,20].flatten())
plt.title("All")
plt.suptitle('Sim Prev NMI')
plt.show()
```


```python

```
