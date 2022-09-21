import os
import argparse
import glob
import pickle
import subprocess
import numpy as np
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from satbenchmark.utils.utils import parse_cnf_file, parse_proof_file, hash_clauses, VIG, write_dimacs_to


class Generator:
    def __init__(self, split, eq_logic):
        self.split = split
        self.eq_logic = eq_logic
        self.exec_dir = os.path.abspath('external/')

    def run(self, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        
        proof_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.proof')
        prover_cmd_line = ['./cadical', '--sat', '--no-binary', cnf_filepath, proof_filepath]

        try:
            process = subprocess.Popen(prover_cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(proof_filepath):
            return

        n_vars, clauses = parse_cnf_file(cnf_filepath)
        
        learned_clauses, deleted_clauses = parse_proof_file(proof_filepath)
        new_clauses = clauses + learned_clauses

        if not self.eq_logic:
            hashed_clauses = hash_clauses(new_clauses)
            hashed_deleted_clauses = hash_clauses(deleted_clauses)

            final_clauses = [new_clauses[idx] for idx, v in enumerate(hashed_clauses) if v not in hashed_deleted_clauses]
        else:
            final_clauses = new_clauses

        vig = VIG(n_vars, clauses)
        assert nx.is_connected(vig)

        formula_filepath = os.path.join(os.path.dirname(os.path.dirname(cnf_filepath)), f'augmented_{self.split}/' + filename + '.cnf')
        write_dimacs_to(n_vars, final_clauses, formula_filepath)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--split', type=str, nargs='+')
    parser.add_argument('--eq_logic', type=bool, default=True, help='Logical equivelent or satisfiability equivelent')
    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    for split in opts.split:
        os.makedirs(os.path.join(opts.input_dir, f'augmented_{split}'), exist_ok=True)

        generater = Generator(split, opts.eq_logic)

        all_files = sorted(glob.glob(opts.input_dir + f'/{split}/*.cnf', recursive=True))
        assert len(all_files) > 0
        all_files = [os.path.abspath(f) for f in all_files]
        
        with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
            pool.map(generater.run, all_files)


if __name__ == '__main__':
    main()
