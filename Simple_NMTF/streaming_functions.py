def overlap(cluster_1,cluster_2,nodes_previous,nodes_current):
    """
    The inputs contain the nodes within the cluster (names of the nodes in str)
    Also need the names of the nodes that were in the previous and current bin.
    """
    numerator_val = len(set(cluster_1).intersection(set(cluster_2)))
    denominator_val = len(((set(cluster_1).union(set(cluster_2))).intersection(set(nodes_previous))).intersection(nodes_current))
    if denominator_val == 0 :
        output_val = 0
    else:
        output_val = numerator_val/denominator_val
    return output_val

def overlap_sets(cluster_set_1,cluster_set_2,nodes_previous,nodes_current):
    """
    These cluster sets are dictionaries where the key corresponds to the cluster number and the value is a set of the node names in the cluster
    This function calculates the overlap between each pair of clusters between the two sets.
    The keys must range from 0 to number of clusters-1
    """
    overlap_matrix = np.zeros((len(cluster_set_1),len(cluster_set_2)))
    for i in range(len(cluster_set_1)):
        for j in range(len(cluster_set_2)):
            overlap_matrix[i,j] = overlap(cluster_set_1[i],cluster_set_2[j],nodes_previous,nodes_current)
    return overlap_matrix
  
  
  
from functools import reduce

def single_node_clustering(total_length,number_changes,number_clusters):
    if number_changes == 0:
        return [np.random.choice(number_clusters)]*total_length
    else:
        times_of_change = np.sort(np.random.choice(np.arange(1,total_length-1).tolist(),size=number_changes,replace=False)+1)
        time_in_each_cluster = np.append(times_of_change,total_length+1)-np.append(1,times_of_change)
        selected_clusters = np.random.choice(number_clusters,number_changes+1)
        cluster_assignment_intermediate = [[selected_clusters[i]]*time_in_each_cluster[i] for i in range(len(time_in_each_cluster))]
        cluster_assignment = []
        for i in range(len(cluster_assignment_intermediate)):
            cluster_assignment = cluster_assignment+cluster_assignment_intermediate[i]
        return cluster_assignment
    
def streaming_cluster_membership(length_of_data,lambda_time_community,number_clusters):
    # length_of_data - how long the changes are occuring over
    # lambda_time_community - vector of lambdas - this allows some nodes to change more than others
    # number_clusters - same number of clusters
    # Get number of clusters change for each node
    number_changes_each = np.random.poisson(lam=lambda_time_community)
    # Membership matrix (rows is each node, column is time)
    membership_matrix = [single_node_clustering(length_of_data,number_changes_each[i],number_clusters) for i in range(len(lambda_time_community))]
    return membership_matrix
