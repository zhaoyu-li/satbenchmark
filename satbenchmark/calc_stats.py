import argparse
import glob
import os
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.community as nx_comm

from networkx.algorithms import bipartite
from satbenchmark.utils.utils import parse_cnf_file, VIG, VCG
from collections import defaultdict


terms = ['n_vars', 'n_clauses', 'vig-diameter', 'vig-characteristic_path_length', \
    'vig-clustering_coefficient', 'vig-modularity', 'vcg-modularity']


def calc_stats(f):
    n_vars, clauses = parse_cnf_file(f)
    vig = VIG(n_vars, clauses)
    vcg = VCG(n_vars, clauses)

    return {
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'vig-diameter': nx.diameter(vig),
        'vig-characteristic_path_length': nx.average_shortest_path_length(vig),
        'vig-clustering_coefficient': nx.average_clustering(vig),
        'vig-modularity': nx_comm.modularity(vig, nx_comm.louvain_communities(vig)),
        'vcg-modularity': nx_comm.modularity(vcg, nx_comm.louvain_communities(vcg))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    parser.add_argument('--split', type=str, choices=[None, 'sat', 'unsat'], default=None, help='SAT/UNSAT')
    opts = parser.parse_args()
    print(opts)
    
    print('Calculating statistics...')
    
    if opts.split is not None:
        all_files = sorted(glob.glob(opts.data_dir + f'/{opts.split}/*.cnf', recursive=True))
    else:
        all_files = sorted(glob.glob(opts.data_dir + '/*/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    stats = defaultdict(int)
    
    for f in tqdm(all_files):
        s = calc_stats(f)
        for t in terms:
            stats[t] += s[t]
    
    for t in terms:
        print('%30s\t%10.2f' % (t, stats[t]/len(all_files)))
    

if __name__ == '__main__':
    main()
