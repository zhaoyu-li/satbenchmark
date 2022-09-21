import argparse


def add_model_options(parser):
    parser.add_argument('--graph', type=str, choices=['lcg', 'vcg'], default='lcg', help='Graph construction')
    parser.add_argument('--init_emb', type=str, choices=['random', 'learned'], default='learned', help='Embedding initialization')
    parser.add_argument('--aggregator', type=str, choices=['sum', 'mean', 'degree-norm'], default='sum', help='Aggregation operator')
    parser.add_argument('--updater', type=str, choices=['gru', 'mlp1', 'mlp2'], default='gru', help='Updating operator')
    
    parser.add_argument('--dim', type=int, default=64, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_iterations', type=int, default=32, help='Number of rounds of message passing')
    
    parser.add_argument('--n_mlp_layers', type=int, default=3, help='Number of layers in all MLPs')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function in all MLPs')
