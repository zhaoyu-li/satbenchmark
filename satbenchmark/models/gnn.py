import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from satbenchmark.models.mlp import MLP
from torch_scatter import scatter_sum, scatter_mean


class GNN(nn.Module):
    def __init__(self, opts):
        super(GNN, self).__init__()
        self.opts = opts
        if self.opts.graph == 'lcg':
            self._lcg_init()
        elif self.opts.graph == 'vcg':
            self._vcg_init()

    def _lcg_init(self):
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.randn(1, self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim))

        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        if self.opts.updater == 'gru':
            self.c_update = nn.GRUCell(self.opts.dim, self.opts.dim)
            self.l_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
        elif self.opts.updater == 'mlp1':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
        elif self.opts.updater == 'mlp2':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 1, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)

        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.init_norm = math.sqrt(self.opts.dim) / nn.init.calculate_gain(self.opts.activation)
    
    def _vcg_init(self):
        if self.opts.init_emb == 'learned':
            self.v_init = nn.Parameter(torch.randn(1, self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim))

        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        if self.opts.updater == 'gru':
            self.c_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
            self.v_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
        elif self.opts.updater == 'mlp1':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
        elif self.opts.updater == 'mlp2':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)

        self.v_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.init_norm = math.sqrt(self.opts.dim) / nn.init.calculate_gain(self.opts.activation)
    
    def forward(self, data):
        if self.opts.graph == 'lcg':
            return self._lcg_forward(data)
        elif self.opts.graph == 'vcg':
            return self._vcg_forward(data)

    def _lcg_forward(self, data):
        batch_size = data.c_size.shape[0]
        batch_size1 = data.num_graphs
        assert batch_size == batch_size1
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1) / self.init_norm
            c_emb = (self.c_init).repeat(c_size, 1) / self.init_norm
        elif self.opts.init_emb == 'random':
            l_emb = torch.randn((l_size, self.opts.dim), device=self.opts.device) / self.init_norm
            c_emb = torch.randn((c_size, self.opts.dim), device=self.opts.device) / self.init_norm

        if self.opts.aggregator == 'mean':
            l_ones = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
            l_deg = scatter_sum(l_ones, l_edge_index, dim=0, dim_size=l_size)
            l_deg[l_deg < 1] = 1
            c_ones = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
            c_deg = scatter_sum(c_ones, c_edge_index, dim=0, dim_size=c_size)
            c_deg[c_deg < 1] = 1
        elif self.opts.aggregator == 'degree-norm':
            l_one = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
            l_deg = scatter_sum(l_one, l_edge_index, dim=0, dim_size=l_size)
            l_deg[l_deg < 1] = 1
            c_one = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
            c_deg = scatter_sum(c_one, c_edge_index, dim=0, dim_size=c_size)
            c_deg[c_deg < 1] = 1
            norm = l_deg[l_edge_index].pow(0.5) * c_deg[c_edge_index].pow(0.5)

        for _ in range(self.opts.n_iterations):
            l2c_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l2c_msg_feat[l_edge_index]
            
            if self.opts.aggregator == 'sum':
                l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            elif self.opts.aggregator == 'mean':
                l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size) / c_deg
            elif self.opts.aggregator == 'degree-norm':
                l2c_msg_aggr = scatter_sum(l2c_msg / norm, c_edge_index, dim=0, dim_size=c_size)
            
            if self.opts.updater == 'gru':
                c_emb = self.c_update(l2c_msg_aggr, c_emb)
            elif self.opts.updater == 'mlp1':
                c_emb = self.c_update(torch.cat([l2c_msg_aggr, c_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                c_emb = self.c_update(l2c_msg_aggr)

            c2l_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c2l_msg_feat[c_edge_index]

            if self.opts.aggregator == 'sum':
                c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            elif self.opts.aggregator == 'mean':
                c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size) / l_deg
            elif self.opts.aggregator == 'degree-norm':
                c2l_msg_aggr = scatter_sum(c2l_msg / norm, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            
            if self.opts.updater == 'gru':
                l_emb = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1), l_emb)
            elif self.opts.updater == 'mlp1':
                l_emb = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg, l_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                l_emb = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1))
        
        if self.opts.task == 'satisfiability':
            l_logit = self.l_readout(l_emb)
            l_batch = data.l_batch
            batch_size = data.num_graphs
            g_logit = scatter_mean(l_logit, l_batch, dim=0, dim_size=batch_size).reshape(-1)
            return F.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            l_logit = self.l_readout(l_emb)
            v_logit = l_logit.reshape(-1, 2)
            return F.softmax(v_logit, dim=1)

    def _vcg_forward(self, data):
        v_size = data.v_size.sum().item()
        c_size = data.c_size.sum().item()

        c_edge_index = data.c_edge_index
        v_edge_index = data.v_edge_index
        p_edge_index = data.p_edge_index
        n_edge_index = data.n_edge_index

        if self.opts.init_emb == 'learned':
            v_emb = (self.v_init).repeat(v_size, 1) / self.init_norm
            c_emb = (self.c_init).repeat(c_size, 1) / self.init_norm
        else:
            v_emb = torch.randn(v_size, self.opts.dim) / self.init_norm
            c_emb = torch.randn(c_size, self.opts.dim) / self.init_norm
        
        if self.opts.aggregator == 'mean':
            p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            p_v_deg[p_v_deg < 1] = 1
            n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            n_v_deg[n_v_deg < 1] = 1

            p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            p_c_deg[p_c_deg < 1] = 1
            n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            n_c_deg[n_c_deg < 1] = 1

        elif self.opts.aggregator == 'degree-norm':
            p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            p_v_deg[p_v_deg < 1] = 1
            n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            n_v_deg[n_v_deg < 1] = 1

            p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            p_c_deg[p_c_deg < 1] = 1
            n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            n_c_deg[n_c_deg < 1] = 1

            p_norm = p_v_deg[v_edge_index[p_edge_index]].pow(0.5) * p_c_deg[c_edge_index[p_edge_index]].pow(0.5)
            n_norm = n_v_deg[v_edge_index[n_edge_index]].pow(0.5) * n_c_deg[c_edge_index[n_edge_index]].pow(0.5)

        for _ in range(self.opts.n_iterations):
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            if self.opts.aggregator == 'sum':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            elif self.opts.aggregator == 'mean':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size) / p_c_deg
            elif self.opts.aggregator == 'degree-norm':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg / p_norm, c_edge_index[p_edge_index], dim=0, dim_size=c_size)

            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            if self.opts.aggregator == 'sum':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            elif self.opts.aggregator == 'mean':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size) / n_c_deg
            elif self.opts.aggregator == 'degree-norm':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg / n_norm, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            
            if self.opts.updater == 'gru':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1), c_emb)
            elif self.opts.updater == 'mlp1':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr, c_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))

            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            if self.opts.aggregator == 'sum':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            elif self.opts.aggregator == 'mean':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size) / p_v_deg
            elif self.opts.aggregator == 'degree-norm':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg / p_norm, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            if self.opts.aggregator == 'sum':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            elif self.opts.aggregator == 'mean':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size) / n_v_deg
            elif self.opts.aggregator == 'degree-norm':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg / n_norm, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            
            if self.opts.updater == 'gru':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1), v_emb)
            elif self.opts.updater == 'mlp1':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr, v_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))            
        
        if self.opts.task == 'satisfiability':
            v_logit = self.v_readout(v_emb)
            v_batch = data.v_batch
            batch_size = data.num_graphs
            g_logit = scatter_mean(v_logit, v_batch, dim=0, dim_size=batch_size).reshape(-1)
            return F.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            v_logit = self.v_readout(v_emb)
            v_prob = F.sigmoid(v_logit)
            return torch.cat([v_prob, 1 - v_prob], dim=1)
