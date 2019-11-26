import sys
sys.path.append(".")
sys.path.append("..")
import os
import datetime
import json
import argparse
from tempfile import mkdtemp
from collections import defaultdict
import subprocess
import math
import torch
from torch import optim
import numpy as np

from utils import Logger, Timer, save_model, save_vars, probe_infnan
import objectives
import models

runId = datetime.datetime.now().isoformat().replace(':','_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

### General
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--model', type=str, metavar='M', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=0, help='print objective values every value (if positive)')
parser.add_argument('--skip-test', action='store_true', default=False, help='skip test dataset computations')

### Dataset
parser.add_argument('--data-params', nargs='+', default=[], help='parameters which are passed to the dataset loader')
parser.add_argument('--data-size', type=int, nargs='+', default=[], help='size/shape of data observations')

### Metric & Plots
parser.add_argument('--iwae-samples', type=int, default=0, help='number of samples to compute marginal log likelihood estimate')

### Optimisation
parser.add_argument('--obj', type=str, default='vae', help='objective to minimise (default: vae)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
parser.add_argument('--lr', type=float, default=1e-4, help='learnign rate for optimser (default: 1e-4)')

## Objective
parser.add_argument('--K', type=int, default=1, metavar='K',  help='number of samples to estimate ELBO (default: 1)')
parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='coefficient of beta-VAE (default: 1.0)')
parser.add_argument('--analytical-kl', action='store_true', default=False, help='analytical kl when possible')

### Model
parser.add_argument('--latent-dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--posterior', type=str, default='WrappedNormal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])

## Architecture
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100, help='number of hidden layers dimensions (default: 100)')
parser.add_argument('--nl', type=str, default='ReLU', help='non linearity')
parser.add_argument('--enc', type=str, default='Wrapped', help='allow to choose different implemented encoder',
                    choices=['Linear', 'Wrapped', 'Mob'])
parser.add_argument('--dec', type=str, default='Wrapped', help='allow to choose different implemented decoder',
                    choices=['Linear', 'Wrapped', 'Geo', 'Mob'])

## Prior
parser.add_argument('--prior-iso', action='store_true', default=False, help='isotropic prior')
parser.add_argument('--prior', type=str, default='WrappedNormal', help='prior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--prior-std', type=float, default=1., help='scale stddev by this value (default:1.)')
parser.add_argument('--learn-prior-std', action='store_true', default=False)

### Technical
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'

# Choosing and saving a random seed for reproducibility
if args.seed == 0: args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
print('seed', args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Create directory for experiment if necessary
directory_name = 'experiments/{}'.format(args.name)
if args.name != '.':
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    runPath = mkdtemp(prefix=runId, dir=directory_name)
else:
    runPath = mkdtemp(prefix=runId, dir=directory_name)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:', runId)

# Save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    fp.write(git_hash.decode('utf-8') + command)
torch.save(args, '{}/args.rar'.format(runPath))

# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
train_loader, test_loader = model.getDataLoaders(args.batch_size, True, device, *args.data_params)
loss_function = getattr(objectives, args.obj + '_objective')


def train(epoch, agg):
    model.train()
    b_loss, b_recon, b_kl = 0., 0., 0.
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        qz_x, px_z, lik, kl, loss = loss_function(model, data, K=args.K, beta=args.beta, components=True, analytical_kl=args.analytical_kl)
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()

        b_loss += loss.item()
        b_recon += -lik.mean(0).sum().item()
        b_kl += kl.sum(-1).mean(0).sum().item()

    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    agg['train_recon'].append(b_recon / len(train_loader.dataset))
    agg['train_kl'].append(b_kl / len(train_loader.dataset))
    if epoch % 1 == 0:
        print('====> Epoch: {:03d} Loss: {:.2f} Recon: {:.2f} KL: {:.2f}'.format(epoch, agg['train_loss'][-1], agg['train_recon'][-1], agg['train_kl'][-1]))


def test(epoch, agg):
    model.eval()
    b_loss, b_mlik = 0., 0.
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            qz_x, px_z, lik, kl, loss = loss_function(model, data, K=args.K, beta=args.beta, components=True)
            if epoch == args.epochs and args.iwae_samples > 0:
                mlik = objectives.iwae_objective(model, data, K=args.iwae_samples)
                b_mlik += mlik.sum(-1).item()
            b_loss += loss.item()
            if i == 0: model.reconstruct(data, runPath, epoch)

    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    agg['test_mlik'].append(b_mlik / len(test_loader.dataset))
    print('====>             Test loss: {:.4f} mlik: {:.4f}'.format(agg['test_loss'][-1], agg['test_mlik'][-1]))


if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        print('Starting training...')

        model.init_last_layer_bias(train_loader)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)
            if args.save_freq == 0 or epoch % args.save_freq == 0:
                if not args.skip_test: test(epoch, agg)
                model.generate(runPath, epoch)
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')

        print('p(z) params:')
        print(model.pz_params)
