import networkx as nx
import torch
import numpy as np
import random

from itertools import combinations


def write_dimacs_to(n_vars, clauses, out_path):
    with open(out_path, 'w') as f:
        f.write('p cnf %d %d\n' % (n_vars, len(clauses)))
        for clause in clauses:
            for literal in clause:
                f.write('%d ' % literal)
            f.write('0\n')


def parse_cnf_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) < 1 or tokens[0] != 'p':
            i += 1
        else:
            break
    
    if i == len(lines):
        return 0, []
    
    header = lines[i].strip().split()
    n_vars = int(header[2])
    n_clauses = int(header[3])
    clauses = []

    for line in lines[i+1:]:
        tokens = line.strip().split()
        if tokens[0] == 'c':
            continue
        clause = [int(s) for s in tokens[:-1]]
        clauses.append(clause)

    return n_vars, clauses


def parse_proof_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    learned_clauses = []
    deleted_clauses = []

    for line in lines:
        tokens = line.strip().split()
        if tokens[0] == 'd':
            deleted_clause = [int(s) for s in tokens[1:-1]]
            deleted_clauses.append(deleted_clause)
        elif len(tokens) > 1: # discard empty clause
            learned_clause = [int(s) for s in tokens[:-1]]
            learned_clauses.append(learned_clause)
    
    return learned_clauses, deleted_clauses


def literal2v_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    return sign, v_idx


def literal2l_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    if sign:
        return v_idx * 2
    else:
        return v_idx * 2 + 1


def VIG(n_vars, clauses):
    G = nx.Graph()
    G.add_nodes_from(range(n_vars))

    for clause in clauses:
        v_idxs = [literal2v_idx(literal)[1] for literal in clause]
        edges = list(combinations(v_idxs, 2))
        G.add_edges_from(edges)
    
    return G


def VCG(n_vars, clauses):
    G = nx.Graph()
    G.add_nodes_from([f'v_{idx}' for idx in range(n_vars)], bipartite=0)
    G.add_nodes_from([f'c_{idx}' for idx in range(len(clauses))], bipartite=1)

    for c_idx, clause in enumerate(clauses):
        edges = [(f'c_{c_idx}', f'v_{literal2v_idx(literal)[1]}') for literal in clause]
        G.add_edges_from(edges)
    
    return G


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def safe_log(t, eps=1e-10):
    t_clamp = t.clamp(min=eps)
    return t_clamp.log()


def hash_clauses(clauses):
    return [hash(frozenset(clause)) for clause in clauses]
