# -*- coding: utf-8 -*-
import logging
import random
import time

import numpy as np
def isin(element, test_elements, assume_unique=False, invert=False):
    "..."
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)

class SBM_bipartite(object):

    def __init__(self, num_vertices_A, num_vertices_B, communities_A, communities_B, vertex_labels_A,vertex_labels_B, p_matrix):
        logging.info('Initializing SBM Model ...')
        self.num_vertices_A = num_vertices_A
        self.num_vertices_B = num_vertices_B
        self.communities_A = communities_A
        self.communities_B = communities_B
        self.vertex_labels_A = vertex_labels_A
        self.vertex_labels_B = vertex_labels_B
        self.p_matrix = p_matrix
        self.block_matrix = self.generate(self.num_vertices_A,self.num_vertices_B, self.communities_A,self.communities_B, self.vertex_labels_A,self.vertex_labels_B, self.p_matrix)

    def detect(self):
        logging.info('SBM detection ...')
        pass

    def generate(self, num_vertices_A, num_vertices_B, num_communities_A, num_communities_B, vertex_labels_A, vertex_labels_B, p_matrix):
        logging.info('Generating SBM (directed graph) ...')
        #v_label_shape = (1, num_vertices)
        p_matrix_shape = (num_communities_A, num_communities_B)
        block_matrix_shape = (num_vertices_A, num_vertices_B)
        block_matrix = np.zeros(block_matrix_shape, dtype=int)
        vertex_labels_A = np.array(vertex_labels_A)
        vertex_labels_B = np.array(vertex_labels_B)
        p_matrix = np.array(p_matrix)

        for row in range(0,num_vertices_A):
            community_a = vertex_labels_A[row]
            community_b = vertex_labels_B
            p = np.random.rand(len(community_b))
            vals = p_matrix[community_a,:][community_b]
            block_matrix[row,:][p<=vals] = 1
            #for col in range(0,num_vertices_B):
            #    community_a = vertex_labels_A[row]
            #    community_b = vertex_labels_B[col]
#
            #    p = random.random()
            #    val = p_matrix[community_a][community_b]

            #    if p <= val:
            #        block_matrix[row][col] = 1



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
        # Now need to check no node has zero degree:
        node_degrees_0 = block_matrix.sum(1)
        node_degrees_1 = block_matrix.sum(0)
        zero_prob_nodes_0 = np.where(p_matrix.max(1)==0)[0]
        zero_prob_nodes_0 = np.where(isin(vertex_labels_A,zero_prob_nodes_0))[0]
        zero_prob_nodes_1 = np.where(p_matrix.max(0)==0)[0]
        zero_prob_nodes_1 = np.where(isin(vertex_labels_B,zero_prob_nodes_1))[0]

        zero_degree_nodes_0 = np.where(node_degrees_0==0)[0]
        zero_degree_nodes_1 = np.where(node_degrees_1==0)[0]
        zero_degree_nodes_0 = np.array(list(set(zero_degree_nodes_0)-set(zero_prob_nodes_0)))
        zero_degree_nodes_1 = np.array(list(set(zero_degree_nodes_1)-set(zero_prob_nodes_1)))
        if (len(zero_degree_nodes_0)==0) & (len(zero_degree_nodes_1)==0):
            zero_nodes = False
        else:
            zero_nodes = True
        iter_number = 0
        if np.any(node_degrees_0==0) | np.any(node_degrees_1==0):
            zero_nodes = True
            stime = time.time()
            while zero_nodes & (iter_number<2):
                zero_degree_nodes_0 = np.where(node_degrees_0==0)[0]
                zero_degree_nodes_1 = np.where(node_degrees_1==0)[0]
                #print(len(zero_prob_nodes_0),len(zero_prob_nodes_1))
                zero_degree_nodes_0 = np.array(list(set(zero_degree_nodes_0)-set(zero_prob_nodes_0)))
                zero_degree_nodes_1 = np.array(list(set(zero_degree_nodes_1)-set(zero_prob_nodes_1)))
                #print(len(zero_degree_nodes_0),len(zero_degree_nodes_1))
                if len(zero_degree_nodes_0)>0:
                    for row in zero_degree_nodes_0:
                        community_a = vertex_labels_A[row]
                        cols = np.array(list(set(range(len(vertex_labels_B)))-set(zero_degree_nodes_1)))
                        community_b = vertex_labels_B[cols]
                        p = np.random.rand(len(community_b))
                        vals = p_matrix[community_a,:][community_b]
                        block_matrix[row,:][cols[p<=vals]] = 1
                        #for col in range(num_vertices_B):
                        #    if not col in zero_degree_nodes_1:
                        #        community_a = vertex_labels_A[row]
                        #        community_b = vertex_labels_B[col]
                        #        p = random.random()
                        #        val = p_matrix[community_a][community_b]
                        #        if p <= val:
                        #            block_matrix[row][col] = 1

                if len(zero_degree_nodes_1)>0:
                    for col in zero_degree_nodes_1:
                        community_b = vertex_labels_B[col]
                        rows = np.array(list(set(range(len(vertex_labels_A)))-set(zero_degree_nodes_0)))
                        community_a = vertex_labels_A[rows]
                        p = np.random.random(len(community_a))
                        vals = p_matrix[:,community_b][community_a]
                        block_matrix[:,col][rows[p<=vals]] = 1
                        #for row in range(num_vertices_A):
                        #    if not row in zero_degree_nodes_0:
                        #        community_a = vertex_labels_A[row]
                        #        community_b = vertex_labels_B[col]
                        #        p = random.random()
                        #        val = p_matrix[community_a][community_b]
                        #        if p <= val:
                        #            block_matrix[row][col] = 1
                if (len(zero_degree_nodes_0)>0) & (len(zero_degree_nodes_1)>0):
                    for row in zero_degree_nodes_0:
                        cols = zero_degree_nodes_1
                        community_a = vertex_labels_A[row]
                        community_b = vertex_labels_B[cols]
                        p = np.random.rand(len(community_b))
                        vals = p_matrix[community_a,:][community_b]
                        block_matrix[row,:][cols[p<=vals]] = 1
                        #for col in zero_degree_nodes_1:
                        #    community_a = vertex_labels_A[row]
                        #    community_b = vertex_labels_B[col]
                        #    p = random.random()
                        #    val = p_matrix[community_a][community_b]
                        #    if p <= val:
                        #        block_matrix[row][col] = 1

                node_degrees_0 = block_matrix.sum(1)
                node_degrees_1 = block_matrix.sum(0)
                iter_number +=1
                zero_degree_nodes_0 = np.where(node_degrees_0==0)[0]
                zero_degree_nodes_1 = np.where(node_degrees_1==0)[0]
                zero_degree_nodes_0 = np.array(list(set(zero_degree_nodes_0)-set(zero_prob_nodes_0)))
                zero_degree_nodes_1 = np.array(list(set(zero_degree_nodes_1)-set(zero_prob_nodes_1)))
                if (len(zero_degree_nodes_0)==0) & (len(zero_degree_nodes_1)==0):
                    zero_nodes = False
        #print("Reruns",time.time()-stime)
        if zero_nodes:
            zero_degree_nodes_0 = np.where(node_degrees_0==0)[0]
            zero_degree_nodes_1 = np.where(node_degrees_1==0)[0]
            # print("No Zero Degree 0:",len(zero_degree_nodes_0))
            # print("No Zero Degree 1:",len(zero_degree_nodes_1))
            for i in zero_degree_nodes_0:
                community_a = vertex_labels_A[i]
                cluster_chosen = np.argmax(p_matrix[community_a,:])
                node_chosen = np.random.choice(np.where(vertex_labels_B == cluster_chosen)[0])
                block_matrix[i][node_chosen] = 1
            for j in zero_degree_nodes_1:
                community_b = vertex_labels_B[j]
                cluster_chosen = np.argmax(p_matrix[:,community_b])
                node_chosen = np.random.choice(np.where(vertex_labels_A == cluster_chosen)[0])
                block_matrix[node_chosen][j] = 1
        return block_matrix

    def recover(self):
        logging.info('SBM recovery ...')
        pass

