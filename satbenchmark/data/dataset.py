import os
import glob
import torch
import pickle
import itertools

from torch_geometric.data import Dataset
from satbenchmark.utils.utils import parse_cnf_file
from satbenchmark.data.lcg import construct_lcg
from satbenchmark.data.vcg import construct_vcg


class SATDataset(Dataset):
    def __init__(self, data_dir, splits, opts):
        self.opts = opts
        self.splits = splits
        self.all_splits, self.all_files = self._get_splits_and_files(data_dir)
        self.all_labels = self._get_labels(data_dir)
            
        super().__init__(data_dir)

    def _get_splits_and_files(self, data_dir):
        all_files = [list(sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True))) for split in self.splits]
        all_files = [[os.path.abspath(cnf_filepath) for cnf_filepath in split_files] for split_files in all_files]
        all_splits = [[split] * len(split_files) for split, split_files in zip(self.splits, all_files)]
        
        return list(itertools.chain(*all_splits)), list(itertools.chain(*all_files)), 
    
    def _get_labels(self, data_dir):
        labels = None
        
        if self.opts.label == 'satisfiability':
            labels = []
            for split, cnf_filepath in zip(self.all_splits, self.all_files):
                if split == 'unsat' or split == 'trimmed' or split == 'augmented_unsat':
                    labels.append(0)
                else:
                    assert split == 'sat' or split == 'augmented_sat'
                    labels.append(1)
            
            labels = [torch.tensor(label, dtype=torch.float) for label in labels]        
        
        elif self.opts.label == 'assignment':
            labels = []
            for cnf_filepath in self.all_files:
                filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_assignment.pkl')
                with open(assignment_file, 'rb') as f:
                    assignment = pickle.load(f)
                labels.append(assignment)

            labels = [torch.tensor(label, dtype=torch.float) for label in labels]
    
        elif self.opts.label == 'core_variable':
            labels = []
            for cnf_filepath in self.all_files:
                filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                core_variable_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_core_variable.pkl')
                with open(core_variable_file, 'rb') as f:
                    core_variable = pickle.load(f)
                labels.append(core_variable)

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
        for split, cnf_filepath in zip(self.all_splits, self.all_files):
            filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
            names.append(f'{split}/{filename}_{self.opts.graph}.pt')

        return names
    
    def process(self):
        for split in self.splits:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)

        for split, cnf_filepath in zip(self.all_splits, self.all_files):
            filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
            name = f'{split}/{filename}_{self.opts.graph}.pt'
            saved_path = os.path.join(self.processed_dir, name)
            if os.path.exists(saved_path):
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            
            if self.opts.graph == 'lcg':
                data = construct_lcg(n_vars, clauses)
            elif self.opts.graph == 'vcg':
                data = construct_vcg(n_vars, clauses)

            torch.save(data, saved_path)
    
    def len(self):
        return len(self.all_files)

    def get(self, idx):
        split = self.all_splits[idx]
        cnf_filepath = self.all_files[idx]
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        name = f'{split}/{filename}_{self.opts.graph}.pt'
        saved_path = os.path.join(self.processed_dir, name)

        data = torch.load(saved_path)
        data.y = self.all_labels[idx]

        return data
