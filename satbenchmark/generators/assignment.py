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
        
        model_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.model')
        cmd_line = ['./cadical', '--sat', cnf_filepath, '-w', model_filepath]

        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(model_filepath):
            return
            
        assignment = []
        with open(model_filepath, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip().split()[1] == 'SATISFIABLE'

            for line in lines[1:]:
                assignment.extend([int(s) for s in line.strip().split()[1:]])
            
            assignment = np.array(assignment[:-1]) > 0 # ends with 0
        
        return assignment

        
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
        assignment = list(pool.map(generater.run, all_files))
    
    with open(os.path.join(opts.input_dir, 'assignment.pkl'), 'wb') as f:
        pickle.dump(assignment, f)


if __name__ == '__main__':
    main()
