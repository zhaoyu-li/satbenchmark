import os
import argparse
import random
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from cnfgen import DominatingSet
from satbenchmark.utils.utils import write_dimacs_to, VIG


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.opts.sat_out_dir = os.path.join(self.opts.out_dir, 'sat')
        self.opts.unsat_out_dir = os.path.join(self.opts.out_dir, 'unsat')
        os.makedirs(self.opts.sat_out_dir, exist_ok=True)
        os.makedirs(self.opts.unsat_out_dir, exist_ok=True)
    
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)
        
        sat = False
        unsat = False
        
        while not sat or not unsat:
            v = random.randint(self.opts.min_v, self.opts.max_v)
            p = 0.25
            graph = nx.generators.erdos_renyi_graph(v, p=p)
            while not nx.is_connected(graph):
                graph = nx.generators.erdos_renyi_graph(v, p=p)
            
            k = random.randint(self.opts.min_k, self.opts.max_k)

            cnf = DominatingSet(graph, k)
            n_vars = len(list(cnf.variables()))
            clauses = list(cnf.clauses())
            clauses = [list(cnf._compress_clause(clause)) for clause in clauses]
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue
            
            solver = Cadical(bootstrap_with=clauses)

            if solver.solve():
                if not sat:
                    sat = True
                    write_dimacs_to(n_vars, clauses, os.path.join(self.opts.sat_out_dir, '%.5d.cnf' % (t)))
            else:
                if not unsat:
                    unsat = True
                    write_dimacs_to(n_vars, clauses, os.path.join(self.opts.unsat_out_dir, '%.5d.cnf' % (t)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_k', type=int, default=3)
    parser.add_argument('--max_k', type=int, default=5)

    parser.add_argument('--min_v', type=int, default=10)
    parser.add_argument('--max_v', type=int, default=40)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    random.seed(opts.seed)

    os.makedirs(opts.out_dir, exist_ok=True)

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))


if __name__ == '__main__':
    main()