# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:00:44 2018

@author: lil115
"""

import pandas as pd
import networkx as nx
import numpy as np
import os
from itertools import chain

def adj2con(adj, n_iter = 10):
    '''
    Given an adjacency matrix, compute the probaility of path containing for each node started at each node.
    The matrix should be at the form of pandas DataFrame.
    '''
    adj.values[[range(len(adj))]*2] = 0
    adj = adj[adj.sum(1)>0]
    
    #adj.values[[range(len(adj))]*2] = adj.apply(lambda x: x[x>0].min(), axis=1) # Assign median edge weights to self loops
    trans = adj.div(adj.sum(1),0)
    temp = pd.DataFrame(np.ones(np.shape(trans)))
    temp.columns = trans.columns
    temp.index = trans.index
    for i in range(n_iter):
        temp.values[[range(len(temp))]*2] = 0
        temp = trans.dot(temp)
    return 1 - temp

def max_isolation(A):
    '''
    Find clusters from A (Pandas DataFrame of contain matrix) by expanding over each gene.
    '''
    
    seeds = A.sum(1).sort_values(ascending=False).index
    A.values[[range(len(A))]*2] = 0 # This change the original input data
    A = (A + A.T) / 2 # This will not change the input data
    A[A<0.01] = 0
    degree = A.sum()
    res = []
    while len(seeds) > 0:
        temp = greedy_expansion(A, [seeds[0]], degree)
        if temp == None: 
            seeds = seeds.drop(seeds[0])
            continue
        clust = set(temp)
        res.append(clust)
        seeds = seeds.drop(seeds[0])
        seeds = seeds.difference(clust)
        print('Remaining seeds: ', len(seeds))
    return merge_overlap(res, A, degree)

def greedy_expansion(A, seed, degree):
    '''
    This method find locally optimal clusters by alternating between shrink and expand.
    '''
    res = set(seed)
    comb = find_candidates(A, res, degree)
    while(True):
        expand = find_candidates1(A, comb, degree)
        if len(expand) == 0: break
        comb = comb.union(expand)
        comb = shrink1(A, comb, degree)
        if len(comb)>200: return None
    return comb

def isolation(A, x, degree):
    return A.loc[x,x].sum().sum() / (2 * degree[x].sum())

def find_candidates(A, res, degree):
    others = A.index.difference(res)
    s = A.loc[res, others].sum() / degree[others]
    return set(s.sort_values(ascending=False).index[:6])
                                                
def find_candidates1(A, res, degree):
    others = A.index.difference(res)
    s = A.loc[res, others].sum() / degree[others]
    crit = isolation(A, res, degree)
    return set(s[s>crit].sort_values(ascending=False).index[:5])

def shrink1(A, res, degree):
    while(True):
        rewards = A.loc[res,res].sum() / degree[res]
        crit = isolation(A, res, degree)
        rewards = set(rewards[rewards<crit].index)
        if len(rewards) == 0: break
        res = res.difference(rewards)
    return res
    
def jaccard(comp_i, comp_j):
    inter = float(len(comp_i & comp_j))
    union = float(len(comp_i | comp_j))
    return inter/union

def eliminate(inds, comp_all, A, degree):
    comps = [comp_all[e] for e in inds]
    comps.append(set.intersection(*comps))
    comps.append(set(chain(*comps)))
    return max(comps, key=lambda x: isolation(A, x, degree))
    
def merge_overlap(comp_all, A, degree, criteria=0.8):
    '''
    Merge components that overlap too much in terms of Jaccard coeffiecients.
    '''
    result = np.zeros((len(comp_all),len(comp_all)))
    for i in range(len(comp_all)):
        result[i,] = [jaccard(comp_all[i], j) for j in comp_all]
    result[result>=criteria] = 1
    result[result<criteria] = 0
    comps_ind = list(nx.connected_components(nx.from_numpy_matrix(result)))
    return [eliminate(e, comp_all, A, degree) for e in comps_ind]

def cut_prefix(se,start_pos=2):
    return {e[start_pos:] for e in se}
                
def remove_small(comp, size=2):
    return [e for e in comp if len(e) > size]

            
### This use case shows how conduct isolation clustering on the three interactomes of yeast in the supplements        

os.chdir('yeast_biogrid/')
graph_names = ['g_perturb_cosine.txt','g_topic_simrank.txt','dig_all_hybrid.txt']
mats = [nx.to_pandas_adjacency(nx.read_weighted_edgelist(e)) for e in graph_names[:2]]
mats.append(nx.to_pandas_adjacency(nx.reverse(nx.read_weighted_edgelist(graph_names[2], create_using=nx.DiGraph()))))

conts = [adj2con(e,5) for e in mats] # Transform the adjacency matrixes to the visiting probability matrix
comps = [max_isolation(e) for e in conts] # Idenify clusters with locally maximal isolation
comps[2] = [cut_prefix(c) for c in comps[2]] # Remove the prefix of the two-layer interactome
comps1 = [remove_small(e,2) for e in comps] # Remove clusters with size smaller than 3.



