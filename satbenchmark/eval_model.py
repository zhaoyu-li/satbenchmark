import torch
import torch.nn.functional as F
import os
import sys
import argparse
import pickle
import time

from satbenchmark.utils.options import add_model_options
from satbenchmark.utils.logger import Logger
from satbenchmark.utils.utils import set_seed, safe_log
from satbenchmark.utils.format_print import FormatTable
from satbenchmark.data.dataloader import get_dataloader
from models.gnn import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment'], help='Experiment task')
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to be tested')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat', 'trimmed'], default=None, help='Directory with validating data')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'core_variable'], default=None, help='Directory with validating data')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)
    
    opts = parser.parse_args()

    set_seed(opts.seed)
    
    opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))
    opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')

    difficulty, dataset = tuple(os.path.abspath(opts.test_dir).split(os.path.sep)[-3:-1])
    checkpoint_name = os.path.splitext(os.path.basename(opts.checkpoint))[0]
    os.makedirs(opts.eval_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)
    test_loader = get_dataloader(opts.test_dir, opts, 'test')

    print('Loading model checkpoint from %s..' % opts.checkpoint)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(opts.checkpoint)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(opts.device)

    all_results = []
    test_tot = 0
    test_acc = 0
    rmse = 0
    solved = 0

    if opts.task == 'satisfiability':
        format_table = FormatTable()

    t0 = time.time()

    print('Testing...')
    model.eval()
    for data in test_loader:
        data = data.to(opts.device)
        batch_size = data.num_graphs
        with torch.no_grad():
            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                loss = F.binary_cross_entropy(pred, label)
                # test_acc += torch.sum((pred > 0.5).float() == label).item()
                format_table.update(pred, label)

                all_results.extend(pred.tolist())
            else:
                pass
            
        test_tot += batch_size
    
    if opts.task == 'satisfiability':
        # test_acc /= test_tot
        # print('Testing accuracy: %f' % test_acc)
        format_table.print_stats()
    elif opts.task == 'assignment':
        pass

    t = time.time() - t0
    print('Solving Time: %f' % t)

    with open('%s/task=%s_difficulty=%s_dataset=%s_split=%s_checkpoint=%s_n_iterations=%d.pkl' % \
        (opts.eval_dir, opts.task, difficulty, dataset, '_'.join(opts.test_splits), checkpoint_name, opts.n_iterations), 'wb') as f:
        pickle.dump((all_results, test_acc), f)


if __name__ == '__main__':
    main()
