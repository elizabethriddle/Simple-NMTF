# -*- coding: utf-8 -*-
import logging
import random

import numpy as np

def isin(element, test_elements, assume_unique=False, invert=False):
    "..."
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)
class SBM(object):

    def __init__(self, num_vertices, communities, vertex_labels, p_matrix):
        logging.info('Initializing SBM Model ...')
        self.num_vertices = num_vertices
        self.communities = communities
        self.vertex_labels = vertex_labels
        self.p_matrix = p_matrix
        self.block_matrix = self.generate(self.num_vertices, self.communities, self.vertex_labels, self.p_matrix)

    def detect(self):
        logging.info('SBM detection ...')
        pass

    def generate(self, num_vertices, num_communities, vertex_labels, p_matrix):
        logging.info('Generating SBM (directed graph) ...')
        v_label_shape = (1, num_vertices)
        p_matrix_shape = (num_communities, num_communities)
        block_matrix_shape = (num_vertices, num_vertices)
        block_matrix = np.zeros(block_matrix_shape, dtype=int)
        p_matrix = np.array(p_matrix)
        vertex_labels = np.array(vertex_labels)

        for row in range(0,num_vertices-1):
            community_a = vertex_labels[row]
            community_b = vertex_labels[range(row+1,num_vertices)]
            p = np.random.rand(len(community_b))
            vals = p_matrix[community_a,:][community_b]
            block_matrix[row,:][np.arange(row+1,num_vertices)[p<=vals]] = 1
            block_matrix[:,row][np.arange(row+1,num_vertices)[p<=vals]] = 1
            # Old version
            #for col in range(row+1,num_vertices):
            #    community_a = vertex_labels[row]
            #    community_b = vertex_labels[col]

            #    p = random.random()
            #    val = p_matrix[community_a][community_b]

            #    if p <= val:
            #        block_matrix[row][col] = 1
            #        block_matrix[col][row] = 1
            
            
            #wrong
            # if sum(block_matrix[row,:])==0:
            #     while sum(block_matrix[row,:])==0:
            #         for col in range(row+1,num_vertices):
            #             community_a = vertex_labels[row]
            #             community_b = vertex_labels[col]
            #
            #             p = random.random()
            #             val = p_matrix[community_a][community_b]
            #
            #             if p <= val:
            #                 block_matrix[row][col] = 1
            #                 block_matrix[col][row] = 1
        
        # Now check that all nodes have connections:
        node_degrees = block_matrix.sum(0)
        zero_prob_nodes = np.where(p_matrix.max(1)==0)[0]
        zero_prob_nodes = np.where(isin(vertex_labels,zero_prob_nodes))[0]
        iter_number = 0
        if np.any(node_degrees==0):
            zero_nodes = True
            while (zero_nodes) & (iter_number<2):
                zero_degree_nodes = np.where(node_degrees==0)[0]
                zero_degree_nodes = np.array(list(set(zero_degree_nodes)-set(zero_prob_nodes)))
                if len(zero_degree_nodes)>0:
                    for i in zero_degree_nodes:
                        vert_positions = np.array(sorted(set(range(num_vertices))-set(zero_degree_nodes)))
                        community_a = vertex_labels[i]
                        community_b = vertex_labels[vert_positions]
                        p = np.random.rand(len(community_b))
                        vals = p_matrix[community_a,:][community_b]
                        block_matrix[i,:][vert_positions[p<=vals]] = 1
                        block_matrix[:,i][vert_positions[p<=vals]] = 1
                    for i in range(len(zero_degree_nodes)-1):
                        position_a = zero_degree_nodes[i]
                        vert_positions = zero_degree_nodes[range(i+1,len(zero_degree_nodes))]
                        community_a = vertex_labels[position_a]
                        community_b = vertex_labels[vert_positions]
                        p = np.random.rand(len(community_b))
                        vals = p_matrix[community_a,:][community_b]
                        block_matrix[position_a,:][vert_positions[p<=vals]] = 1
                        block_matrix[:,position_a][vert_positions[p<=vals]] = 1
                node_degrees = block_matrix.sum(0)
                zero_degree_nodes = np.where(node_degrees==0)[0]
                zero_degree_nodes = np.array(list(set(zero_degree_nodes)-set(zero_prob_nodes)))
                iter_number +=1
                if not np.any(node_degrees==0):
                    zero_nodes = False
        if np.any(node_degrees==0):
            zero_degree_nodes = np.where(node_degrees==0)[0]
            #print("No Zero Degree:",len(zero_degree_nodes))
            for i in zero_degree_nodes:
                community_a = vertex_labels[i]
                cluster_chosen = np.argmax(p_matrix[community_a,:])
                node_chosen = np.random.choice(np.where(vertex_labels == cluster_chosen)[0])
                block_matrix[i][node_chosen] = 1
                block_matrix[node_chosen][i] = 1
                               
        return block_matrix

    def recover(self):
        logging.info('SBM recovery ...')
        pass

    
    
    
    
    



