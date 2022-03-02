# Simple-NMTF
Fast multi-relational clustering procedure to group similar entities.

## Abstract

Several cyber-security data sources are collected in enterprise networks providing relational information between different types of nodes in the network, namely computers, users and ports. This relational data can be expressed as adjacency matrices detailing inter-type relationships corresponding to relations between nodes of different types and intra-type relationships showing relationships between nodes of the same type. In this paper, we propose an extension of Non-Negative Matrix Tri-Factorisation (NMTF) to simultaneously cluster nodes based on their intra and inter-type relationships. Existing NMTF based clustering methods suffer from long computational times due to large matrix multiplications. In our approach, we enforce stricter cluster indicator constraints on the factor matrices to circumvent these issues. Additionally, to make our proposed approach less susceptible to variation in results due to random initialisation, we propose a novel initialisation procedure based on Non-Negative Double Singular Value Decomposition for multi-type relational clustering. Finally, a new performance measure suitable for assessing clustering performance on unlabelled multi-type relational data sets is presented. Our algorithm is assessed on both a simulated and real computer network against standard approaches showing its strong performance.

For full details of this work see:

Riddle-Workman, Elizabeth,  Evangelou, Marina, & Adams,  Niall  M.  (2021).   Multi-Type Relational Clustering for Enterprise Cyber-Security Networks.  Pattern Recognition Letters. https://www.sciencedirect.com/science/article/abs/pii/S0167865521002051?via%3Dihub
