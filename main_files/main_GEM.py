#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be

from gem.utils import graph_util
from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE

import argparse
import numpy as np
import networkx as nx
import ast


def parse_args():
    """ Parses arguments."""

    parser = argparse.ArgumentParser(description="Run GEM method.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb.csv',
                        help='Embeddings path. Default is `emb.csv`')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 128.')

    parser.add_argument('--method', required=True, choices=['lle', 'hope', 'lap', 'gf', 'sdne'],
                        help='The network embedding method')

    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as weighted')

    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')

    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations for gf and sdne. Default is 10.')

    parser.add_argument('--eta', type=float, default=1*10**-4,
                        help='Learning rate parameter for gf. Default is 1*10**-4.')

    parser.add_argument('--regu', type=float, default=1.0,
                        help='Regularization parameter for gf. Default is 1.0.')

    parser.add_argument('--alpha', type=float, default=1e-5,
                        help='First order proximity weight for SDNE. Default is 1e-5.')

    parser.add_argument('--beta', type=float, default=0.01,
                        help='Beta parameter for SDNE and HOPE. Default is 0.01.')

    parser.add_argument('--nu1', default=1e-6, type=float,
                        help='Lasso regularization coefficient for SDNE. Default is 1e-6.')

    parser.add_argument('--nu2', default=1e-6, type=float,
                        help='Ridge regression coefficient for SDNE. Default is 1e-6.')

    parser.add_argument('--bs', default=500, type=int,
                        help='Batch size parameter for SDNE. Default is 500.')

    parser.add_argument('--encoder-list', default='[500, 128]', type=str,
                        help='A list of numbers of the neuron at each encoder layer for SDNE, the last number is the '
                             'dimension of the output node representation. Default is [500, 128].')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for SDNE. Default is 0.01.')

    return parser.parse_args()


def prep_graph(G, relabel=True, del_self_loops=True):
    r"""
    Preprocess a graphs according to the parameters provided.
    By default the (digraphs) graphs are restricted to their main (weakly) connected component.
    Trying to embed graphs with several CCs may cause some algorithms to put them infinitely far away.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    relabel : bool, optional
       Determines if the nodes are relabeled with consecutive integers 0..N
    del_self_loops : bool, optional
       Determines if self loops should be deleted from the graph. Default is True.

    Returns
    -------
    G : graph
       A preprocessed NetworkX graph
    Ids : list of tuples
       A list of (OldNodeID, NewNodeID)
    """
    # Remove self loops
    if del_self_loops:
        G.remove_edges_from(G.selfloop_edges())

    # Restrict graph to its main connected component
    if G.is_directed():
        Gcc = max(nx.weakly_connected_component_subgraphs(G), key=len)
    else:
        Gcc = max(nx.connected_component_subgraphs(G), key=len)

    # Relabel graph nodes in 0...N
    if relabel:
        Grl = nx.convert_node_labels_to_integers(Gcc, first_label=0, ordering='sorted')
        # A list of (oldNodeID, newNodeID)
        ids = list(zip(sorted(Gcc.nodes), sorted(Grl.nodes)))
        return Grl, ids
    else:
        return Gcc, None


def main(args):

    # Load edgelist
    G = graph_util.loadGraphFromEdgeListTxt(args.input, directed=args.directed)
    G = G.to_directed()

    # Preprocess the graph
    # G, _ = prep_graph(G)

    if args.method == 'gf':
        # GF takes embedding dimension (d), maximum iterations (max_iter), learning rate (eta),
        # regularization coefficient (regu) as inputs
        model = GraphFactorization(d=args.dimension, max_iter=args.max_iter, eta=args.eta, regu=args.regu)
    elif args.method == 'hope':
        # HOPE takes embedding dimension (d) and decay factor (beta) as inputs
        model = HOPE(d=args.dimension, beta=args.beta)
    elif args.method == 'lap':
        # LE takes embedding dimension (d) as input
        model = LaplacianEigenmaps(d=args.dimension)
    elif args.method == 'lle':
        # LLE takes embedding dimension (d) as input
        model = LocallyLinearEmbedding(d=args.dimension)
    elif args.method == 'sdne':
        encoder_layer_list = ast.literal_eval(args.encoder_list)
        # SDNE takes embedding dimension (d), seen edge reconstruction weight (beta), first order proximity weight
        # (alpha), lasso regularization coefficient (nu1), ridge regreesion coefficient (nu2), number of hidden layers
        # (K), size of each layer (n_units), number of iterations (n_ite), learning rate (xeta), size of batch (n_batch)
        # location of modelfile and weightfile save (modelfile and weightfile) as inputs
        model = SDNE(d=args.dimension, beta=args.beta, alpha=args.alpha, nu1=args.nu1, nu2=args.nu2,
                     K=len(encoder_layer_list), n_units=encoder_layer_list,
                     n_iter=args.max_iter, xeta=args.learning_rate, n_batch=args.bs)
        # , modelfile=['enc_model.json', 'dec_model.json'], weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])
    else:
        raise ValueError('The requested method does not exist!')

    # Learn the node embeddings
    Y, t = model.learn_embedding(graph=G, edge_f=None, is_weighted=args.weighted, no_python=True)
    Z = np.real_if_close(Y, tol=1000)

    # Save the node embeddings to a file
    np.savetxt(args.output, Z, delimiter=',', fmt='%f')


if __name__ == "__main__":
    args = parse_args()
    main(args)
