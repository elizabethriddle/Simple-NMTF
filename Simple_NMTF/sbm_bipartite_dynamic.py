# -*- coding: utf-8 -*-
import logging
import random
import time

import numpy as np


class SBM_dynamic(object):

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
        
                               
        return block_matrix

    def recover(self):
        logging.info('SBM recovery ...')
        pass

    

class SBM_bipartite_dynamic(object):

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

    def generate(self, num_vertices_A,num_vertices_B, num_communities_A,num_communities_B, vertex_labels_A,vertex_labels_B, p_matrix):
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
        

        return block_matrix

    def recover(self):
        logging.info('SBM recovery ...')
        pass