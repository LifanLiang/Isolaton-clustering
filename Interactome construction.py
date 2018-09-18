# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:56:57 2018

@author: lil115
"""

import pandas as pd
import numpy as np
import networkx as nx


def cosine_sim(A):
    similarity = np.dot(A, A.T)
    
    square_mag = np.diag(similarity)
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    
    cosine = (inv_mag * similarity).T * inv_mag
    return cosine

def simrank(deg):
    gene_to_condition = deg/deg.sum(0)
    condition_to_gene = deg.T/deg.T.sum(0)
    
    #s_cur = pd.DataFrame(cosine_sim(deg.T))
    s_cur = pd.DataFrame(np.zeros((len(deg.columns),len(deg.columns))))
    s_cur.index = deg.columns
    s_cur.columns = deg.columns
    c = 0.8
    s_prev = s_cur.copy()
    for i in range(100):
        s_cond = c * condition_to_gene.T.dot(s_prev).dot(condition_to_gene)
        s_cond.values[[np.arange(len(deg))]*2] = 1.0
        s_cur = c * gene_to_condition.T.dot(s_cond).dot(gene_to_condition)
        s_cur.values[[np.arange(len(deg.T))]*2] = 1.0
        if i % 3 == 0:
            if np.allclose(s_prev,s_cur,atol=1e-10): break
        s_prev = s_cur.copy()
    return s_cur

def filter_single_nodes(G):
    G.remove_edges_from([(u,v) for u,v in G.edges() if G[u][v]['weight']==0])
    pert_deg = G.degree()
    G.remove_nodes_from([e for e in pert_deg if pert_deg[e] == 0])

def find_common_nodes(g1,g2):
    filter_single_nodes(g1)
    filter_single_nodes(g2)
    commgenes = list(set.intersection(set(g1.nodes()),set(g2.nodes()))) 
    while g1.nodes() != g2.nodes():
        g1 = g1.subgraph(commgenes)
        g2 = g2.subgraph(commgenes)
        filter_single_nodes(g1)
        filter_single_nodes(g2)
        commgenes = list(set.intersection(set(g1.nodes()),set(g2.nodes()))) 
    return g1,g2
    
g_ppi = nx.read_edgelist('yeast_biogrid.edg') # The edgelist of 
topic = pd.read_csv('20160323.Pubmed.TfIdfMin53Kept6000_Output-Run-8_count_matrix-labeled.csv',index_col=0)
deg_min5 = pd.read_csv('DEG_samples1784_min5_Trans.csv', index_col=0)
coex_min5_cos = pd.DataFrame(cosine_sim(deg_min5.T))

## construct single-layer interactome annotated by coexpression
genes = deg_min5.columns
g_perturb_min5_cosine = nx.from_edgelist([(u,v,{'weight': coex_min5_cos.loc[u,v]}) for u,v in g_ppi.edges() if ((u in genes) and (v in genes))])
g_perturb_min5_cosine.remove_edges_from([(u,v) for u,v in g_perturb_min5_cosine.edges() if g_perturb_min5_cosine[u][v]['weight']==0])
pert_deg = g_perturb_min5_cosine.degree()
g_perturb_min5_cosine.remove_nodes_from([e for e in pert_deg if pert_deg[e] > 500]) # Remove too central nodes to reveal the network topology better
pert_deg = g_perturb_min5_cosine.degree()
g_perturb_min5_cosine.remove_nodes_from([e for e in pert_deg if pert_deg[e] == 0])

## Preprocessing of Topic-Gene association matrix
topic.fillna(0)
topic = topic[topic.sum()[topic.sum()>0].index]
topic_round = topic.applymap(lambda x: 0 if x<1 else x)
topic_sim = simrank(topic_round)

## Construct single-layer interactome of topic gene association
genes_topic = topic_sim.index
g_topic_simrank = nx.from_edgelist([(u,v,{'weight': topic_sim.loc[u,v]}) for u,v in g_ppi.edges() if ((u in genes_topic) and (v in genes_topic))])
g_topic_simrank.remove_edges_from([(u,v) for u,v in g_topic_simrank.edges() if g_topic_simrank[u][v]['weight']==0])
pert_deg = g_topic_simrank.degree()
g_topic_simrank.remove_nodes_from([e for e in pert_deg if pert_deg[e] == 0])

## Construct the combined interactome
commgenes = list(set.intersection(set(g_ppi.nodes()),set(g_perturb_min5_cosine.nodes()),set(g_topic_simrank.nodes())))
g1, g2 = find_common_nodes(g_perturb_min5_cosine.subgraph(commgenes),g_topic_simrank.subgraph(commgenes))
g_all_min5_hybrid = nx.union(g1,g2,rename=('p-','t-'))
dig_all_min5_hybrid = nx.DiGraph(g_all_min5_hybrid)
t_node = ['t-'+e for e in g1.nodes()]
p_node = ['p-'+e for e in g2.nodes()]
outdeg_all = dig_all_min5_hybrid.out_degree()
dig_all_min5_hybrid.add_weighted_edges_from([(t_node[i],p_node[i], dig_all_min5_hybrid.out_degree(t_node[i],weight='weight')/outdeg_all[t_node[i]]) for i in range(len(t_node))])
dig_all_min5_hybrid.add_weighted_edges_from([(p_node[i],t_node[i], dig_all_min5_hybrid.out_degree(p_node[i],weight='weight')/outdeg_all[p_node[i]]) for i in range(len(p_node))])
