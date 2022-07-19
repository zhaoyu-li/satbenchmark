import os
import argparse
import glob
import pickle
import subprocess
import numpy as np

from concurrent.futures.process import ProcessPoolExecutor


class Generator:
    def __init__(self):
        self.exec_dir = os.path.abspath('external/')

    def run(self, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        
        proof_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.proof')
        prover_cmd_line = ['./cadical', '--unsat', cnf_filepath, proof_filepath]

        try:
            process = subprocess.Popen(prover_cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(proof_filepath):
            return

        core_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.core')
        checker_cmd_line = ['./drat-trim', cnf_filepath, proof_filepath, '-c', core_filepath]

        try:
            process = subprocess.Popen(checker_cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(core_filepath):
            return

        with open(core_filepath, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split()
            n_vars = int(header[2])
            n_clauses = int(header[3])

            unsat_core = np.zeros(n_vars)
            for line in lines[1:]:
                tokens = line.strip().split()
                unsat_core[[abs(int(t))-1 for t in tokens[:-1]]] = 1
        
        return unsat_core

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator()

    all_files = sorted(glob.glob(opts.input_dir + '/*.cnf', recursive=True))
    assert len(all_files) > 0
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        unsat_core = list(pool.map(generater.run, all_files))
    
    with open(os.path.join(opts.input_dir, 'unsat_core.pkl'), 'wb') as f:
        pickle.dump(unsat_core, f)


if __name__ == '__main__':
    main()
