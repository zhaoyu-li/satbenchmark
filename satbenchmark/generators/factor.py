import os
import argparse
import random
import sympy
import subprocess
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from satbenchmark.utils.utils import parse_cnf_file, write_dimacs_to, VIG


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.opts.sat_out_dir = os.path.join(self.opts.out_dir, 'sat')
        self.exec_dir = os.path.abspath('external/')
        os.makedirs(self.opts.sat_out_dir, exist_ok=True)
        
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)
        
        while True:
            n_factors = random.randint(self.opts.min_f, self.opts.max_f)
            factors = 1
            for i in range(n_factors):
                factors *= sympy.randprime(self.opts.min_p, self.opts.max_p)
            bitstr = bin(factors)[2:]
            
            cnf_filepath = os.path.join(self.opts.out_dir, '%.5d.cnf' % (t))
            factor_cmd_line = ['./satfactor', bitsstr]

            with open(cnf_filepath, 'w') as f:
                try:
                    process = subprocess.Popen(factor_cmd_line, stdout=f, stderr=f, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(cnf_filepath).st_size == 0:
                os.remove(cnf_filepath)
                continue
            
            simplifier_cmd_line = ['./satelite', cnf_filepath, cnf_filepath]

            try:
                process = subprocess.Popen(simplifier_cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(cnf_filepath).st_size == 0:
                os.remove(cnf_filepath)
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            vig = VIG(n_vars, clauses)
            if nx.is_connected(vig):
                break

        solver = Cadical(bootstrap_with=clauses)
        sat = solver.solve()
        assert sat == True
        write_dimacs_to(n_vars, clauses, os.path.join(self.opts.sat_out_dir, '%.5d.cnf' % (t)))
        os.remove(cnf_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_p', type=float, default=1000)
    parser.add_argument('--max_p', type=float, default=100000)

    parser.add_argument('--min_f', type=float, default=3)
    parser.add_argument('--max_f', type=float, default=5)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))
    

if __name__ == '__main__':
    main()
