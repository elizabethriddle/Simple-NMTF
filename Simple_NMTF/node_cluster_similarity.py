def node_cluster_similarity(A,cluster_1,cluster_2):
    """
    Calculate node cluster similarity over a matrix (biclustering).
    :param A: adjacency matrix 
    :param cluster_1: clustering for the rows
    :param cluster_2: clustering for the columns
    :return: the biclustering performance over matrix A
    """
    ## Gives sum of within cluster similarity (when combining or calculating final value divide by number of nodes only)
    
    # First need to calculate the similarity of edges between nodes in each cluster:
    # Find within cluster similarity:
    cluster_1 = np.array(cluster_1)
    cluster_2 = np.array(cluster_2)
    A = lil_matrix(A)
    # Caculate sum within cluster edge sims:
    within_cluster_edge_sim = []
    for x in range(len(set(cluster_1))):
        cosine_sims = cosine_similarity(A[cluster_1==x,:])
        if np.sum(cluster_1==x)>1:
            within_cluster_edge_sim.append(np.sum(cosine_sims[np.triu_indices(cosine_sims.shape[0],1)]))
        else:
            within_cluster_edge_sim.append(1)
    
    # Calculate sum within cluster cluster sims:
    P = np.zeros((A.shape[1],len(set(cluster_2))))
    P[np.arange(P.shape[0]),cluster_2] = 1
    P = lil_matrix(P)
    A_cluster = A.dot(P)
    within_cluster_cluster_sim = []
    for x in range(len(set(cluster_1))):
        cosine_sims = cosine_similarity(A_cluster[cluster_1==x,:])
        if np.sum(cluster_1==x)>1:
            within_cluster_cluster_sim.append(np.sum(cosine_sims[np.triu_indices(cosine_sims.shape[0],1)]))
        else:
            within_cluster_cluster_sim.append(1)
    # Take average of two similarity measures
    within_cluster_sim = (np.array(within_cluster_edge_sim)+np.array(within_cluster_cluster_sim))*0.5
    
    # Now divide by C_k-1:
    # First get cluster sizes:
    uniq_cluster_no,cluster_sizes = np.unique(cluster_1,return_counts=True)
    weight_within_cluster_sim = within_cluster_sim/(cluster_sizes-1)
    weight_within_cluster_sim[np.isinf(weight_within_cluster_sim)]=1/2
    return weight_within_cluster_sim.sum()*2
