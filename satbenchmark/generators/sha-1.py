import os
import argparse
import random
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
    
    def random_binary_string(self, n):
        return ''.join([str(random.randint(0, 1)) for _ in range(n)])
        
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)
        
        while True:
            n_rounds = 17
            n_bits = random.randint(self.opts.min_b, self.opts.max_b)

            bitsstr = '0b'+self.random_binary_string(512)
            
            cnf_filepath = os.path.join(self.opts.out_dir, '%.5d.cnf' % (t))
            cmd_line = ['./cgen', 'encode', 'SHA1', '-vM', bitsstr, 'except:1..'+str(n_bits), \
                 '-vH', 'compute', '-r', str(n_rounds), cnf_filepath]
            
            try:
                process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if not os.path.exists(cnf_filepath):
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

    parser.add_argument('--min_b', type=float, default=20)
    parser.add_argument('--max_b', type=float, default=40)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))
    

if __name__ == '__main__':
    main()
