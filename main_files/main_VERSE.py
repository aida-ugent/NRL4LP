#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be

import argparse
import numpy as np
import networkx as nx
import subprocess
import os
from convert import *

def parse_args():
    """ Parses Verse arguments. """

    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--input', nargs='?',
                        default='../data/test.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?',
                        default='../data/network.emb',
                        help='Output embedding path')

    parser.add_argument('--dimension', type=int, default=8,
                        help='Embedding dimension. Default is 8.')

    parser.add_argument('--format', default='edgelist',
                        help='Format of the input file. Options are: [`weighted_edgelist`, `edgelist`, `adjlist`]. Default is `edgelist`')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in input file. Default is `,`')

    parser.add_argument('--alpha', type=float, default=0.85,
                        help='Learning rate. Default is 0.85.')

    parser.add_argument('--threads', type=int, default=4,
                        help='The number of parallel threads. Default is 4.')

    parser.add_argument('--nsamples', type=int, default=3,
                        help='Numer of negative samples. Default is 3.')
    
    parser.add_argument('--directed', dest='directed', action='store_true',
	                help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def load_embeddings(file_name, dimensions):
    """ Loads method embeddings. """
 
    embeddings = np.fromfile(file_name, dtype=np.float32)
    length = embeddings.shape[0]
    assert length % dimensions == 0
    embedding_shape = [int(length / dimensions), dimensions]
    embeddings = embeddings.reshape(embedding_shape)
    return embeddings


def main(args):
    """ Compute embeddings using Verse c++ code. """

    # Transform the input to bcsr
    if args.format in ['weighted_edgelist', 'edgelist', 'adjlist']:
        indptr, indices, weights = list2mat(args.input, args.directed, args.delimiter, args.format)

    with open(args.output, 'wb') as fout:
        xgfs2file(fout, indptr, indices, weights)

    # Get the verse full path relative to this main file path
    aux = os.path.realpath(__file__)
    path = os.path.dirname(aux)

    # Call the c++ implementation
    command = path + "/../src/verse -input {} -output {} -dim {} -alpha {} -threads {} -nsamples {}".format(args.output, args.output, args.dimension, args.alpha, args.threads, args.nsamples)
    subprocess.call(command, shell=True)
    
    # Read output and transform it to matrix
    #e = Embedding(args.output, args.dimension)
    e = load_embeddings(args.output, args.dimension)

    # Save the embeddings in a decent format
    np.savetxt(args.output, e, delimiter=args.delimiter)


if __name__ == "__main__":
    args = parse_args()
    main(args)
