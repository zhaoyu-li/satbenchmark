import os
import glob
import torch
import pickle
import itertools
import numpy as np

from torch_geometric.data import Dataset
from satbenchmark.utils.utils import parse_cnf_file
from satbenchmark.data.lcg import construct_lcg
from satbenchmark.data.vcg import construct_vcg


class SATDataset(Dataset):
    def __init__(self, data_dir, opts):
        self.opts = opts
        if self.opts.split is not None:
            all_files = sorted(glob.glob(data_dir + f'/{self.opts.split}/*.cnf', recursive=True))
        else:
            all_files = sorted(glob.glob(data_dir + '/*/*.cnf', recursive=True))

        self.all_files = [os.path.abspath(f) for f in all_files]
        self.all_labels = self._get_labels(data_dir)
            
        super().__init__(data_dir)
    
    def _get_labels(self, data_dir):
        labels = None
        
        if self.opts.label == 'satisfiability':
            labels = []
            for f in self.all_files:
                if 'unsat' in f:
                    labels.append(0)
                else:
                    labels.append(1)
            labels = [torch.tensor(label, dtype=torch.float) for label in labels]        
        elif self.opts.label == 'assignment':
            assert self.opts.split == 'sat'
            labels_file = os.path.join(data_dir, 'sat/assignments.pkl')
            if os.path.exists(labels_file):
                with open(labels_file, 'rb') as f:
                    labels = pickle.load(f)
                labels = [torch.tensor(label, dtype=torch.float) for label in labels]        
        elif self.opts.label == 'unsat_core':
            assert self.opts.split == 'unsat'
            labels_file = os.path.join(data_dir, 'unsat/unsat_core.pkl')
            if os.path.exists(labels_file):
                with open(labels_file, 'rb') as f:
                    labels = pickle.load(f)
                labels = [torch.tensor(label, dtype=torch.float) for label in labels]
        
        if labels is None:
            labels = [None] * len(self.all_files)

        assert len(labels) == len(self.all_files)
        
        return labels
    
    @property
    def processed_file_names(self):
        sat_idx = 0
        unsat_idx = 0
        
        names = []
        for f in self.all_files:
            if 'unsat' in f:
                names.append(f'unsat/data_{unsat_idx}_{self.opts.graph}.pt')
                unsat_idx += 1
            else:
                names.append(f'sat/data_{sat_idx}_{self.opts.graph}.pt')
                sat_idx += 1

        return names
    
    def process(self):
        os.makedirs(os.path.join(self.processed_dir, 'sat'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'unsat'), exist_ok=True)

        sat_idx = 0
        unsat_idx = 0

        for f in self.all_files:
            n_vars, clauses = parse_cnf_file(f)
            
            if self.opts.graph == 'lcg':
                data = construct_lcg(n_vars, clauses)
            elif self.opts.graph == 'vcg':
                data = construct_vcg(n_vars, clauses)
            
            if 'unsat' in f:
                file_name = f'unsat/data_{unsat_idx}_{self.opts.graph}.pt'
                unsat_idx += 1
            else:
                file_name = f'sat/data_{sat_idx}_{self.opts.graph}.pt'
                sat_idx += 1
            
            torch.save(data, os.path.join(self.processed_dir, file_name))
    
    def len(self):
        return len(self.all_files)

    def get(self, idx):
        if self.opts.split is not None:
            file_name = f'/{self.opts.split}/data_{idx}_{self.opts.graph}.pt'
        else:
            if idx < len(self.all_files)/2:
                file_name = f'sat/data_{idx}_{self.opts.graph}.pt'
            else:
                file_name = f'unsat/data_{idx-len(self.all_files)//2}_{self.opts.graph}.pt'
            
        data = torch.load(os.path.join(self.processed_dir, file_name))
        data.y = self.all_labels[idx]
        return data
