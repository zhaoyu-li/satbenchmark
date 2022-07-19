import torch

from torch_geometric.data import Data
from satbenchmark.utils.utils import literal2l_idx


class LCG(Data):
    def __init__(self,
            l_size=None,
            c_size=None,
            l_edge_index=None,
            c_edge_index=None,
            l_batch=None,
            c_batch=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.c_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l_edge_index':
            return self.l_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'l_batch' or key == 'c_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


def construct_lcg(n_vars, clauses):
    l_edge_index_list = []
    c_edge_index_list = []
    
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            l_idx = literal2l_idx(literal)
            l_edge_index_list.append(l_idx)
            c_edge_index_list.append(c_idx)
    
    return LCG(
        n_vars * 2,
        len(clauses),
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars * 2, dtype=torch.long),
        torch.zeros(len(clauses), dtype=torch.long)
    )
