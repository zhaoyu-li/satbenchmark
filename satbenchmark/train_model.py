import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from satbenchmark.utils.options import add_model_options
from satbenchmark.utils.utils import set_seed, safe_log
from satbenchmark.utils.logger import Logger
from satbenchmark.utils.format_print import FormatTable
from satbenchmark.data.dataloader import get_dataloader
from satbenchmark.models.gnn import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment'], help='Experiment task')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat', 'trimmed'], default=None, help='Category of the training data')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat', 'trimmed'], default=None, help='Category of the validating data')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'unsat_core'], default=None, help='Directory with validating data')
    parser.add_argument('--loss', type=str, choices=[None, 'unsupervised1', 'unsupervised2', 'supervised'], default=None, help='Loss type for assignment prediction')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs during training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_dacay', type=float, default=1e-6, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=200, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=0.65, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)

    opts = parser.parse_args()

    set_seed(opts.seed)
    difficulty, dataset = tuple(os.path.abspath(opts.train_dir).split(os.path.sep)[-3:-1])
    splits_name = '_'.join(opts.train_splits)
    exp_name = f'task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_label={opts.label}_loss={opts.loss}/' + \
        f'graph={opts.graph}_init_emb={opts.init_emb}_aggregator={opts.aggregator}_updater={opts.updater}_n_iterations={opts.n_iterations}'

    opts.log_dir = os.path.join('runs', exp_name)
    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_dacay)
    train_loader = get_dataloader(opts.train_dir, opts.train_splits, opts, 'train')
    
    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts.valid_splits, opts, 'valid')
    else:
        valid_loader = None
    
    if opts.scheduler is not None:
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)
    
    if opts.task == 'satisfiability':
        format_table = FormatTable()

    best_loss = float('inf')

    for epoch in range(opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = 0
        train_acc = 0
        train_tot = 0

        if opts.task == 'satisfiability':
            format_table.reset()
        
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(opts.device)
            batch_size = data.num_graphs
            
            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                loss = F.binary_cross_entropy(pred, label)
                # train_acc += torch.sum((pred > 0.5).float() == label).item()
                format_table.update(pred, label)
            else:
                pass

            train_loss += loss.item() * batch_size
            train_tot += batch_size
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
            optimizer.step()
            
        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if opts.task == 'satisfiability':
            # train_acc /= train_tot
            format_table.print_stats()
            # print('Training accuracy: %f' % train_acc)
        elif opts.task == 'assignment':
            pass

        if epoch % opts.save_model_epochs == 0:
            torch.save({
                'state_dict': model.state_dict(), 
                'epoch': epoch,
                'optimizer': optimizer.state_dict()}, 
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )
        
        if opts.valid_dir is not None:
            print('Validating...')

            valid_loss = 0
            valid_acc = 0
            valid_tot = 0

            if opts.task == 'satisfiability':
                format_table.reset()
            
            model.eval()
            for data in valid_loader:
                data = data.to(opts.device)
                batch_size = data.num_graphs
                with torch.no_grad():
                    if opts.task == 'satisfiability':
                        pred = model(data)
                        label = data.y
                        loss = F.binary_cross_entropy(pred, label)

                        format_table.update(pred, label)
                        # valid_acc += torch.sum((pred > 0.5).float() == label).item()
                    else:
                        pass
                                
                valid_loss += loss.item() * batch_size
                valid_tot += batch_size
            
            valid_loss /= valid_tot
            print('Validating loss: %f' % valid_loss)

            if opts.task == 'satisfiability':
                # valid_acc /= valid_tot
                format_table.print_stats()
                # print('Validating accuracy: %f' % valid_acc)
            elif opts.task == 'assignment':
                pass

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch, 
                    'optimizer': optimizer.state_dict()}, 
                    os.path.join(opts.checkpoint_dir, 'model_best.pt')
                )

            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
        else:
            if opts.scheduler is not None:
                scheduler.step()


if __name__ == '__main__':
    main()
