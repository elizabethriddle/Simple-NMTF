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

