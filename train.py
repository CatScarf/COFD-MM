import logging
import os
import time
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torchvision
from torch import Tensor, nn
from torch.nn.utils import rnn
from tqdm import tqdm

import model
import read_data as data
import util
from config import Args, get_args
from metrics import MetricsAll, merge_metrics

real_dir = os.path.dirname(__file__)
BATCH = Tuple[Tensor, Tensor, Tensor, Tensor, List[int], Tensor, Tensor]
def train_encoder(
    model: model.Encoder,
    batchs: Union[List[BATCH], Iterator[BATCH]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metrics: MetricsAll,
    optimizer: torch.optim.Optimizer,
    args: Args,
):

    srcs: List[Tensor] = []
    embs: Optional[List[Tensor]] = []
    for batch in batchs:
        # Get data.
        src, dst, nfeat, efeat, sections, mask, label = batch
        train_mask = mask == 0
        def to_device(x: Tensor):
            try:
                return x.to(args.device)
            except:
                return x.cuda()
        nfeat, efeat, mask, label = map(to_device, [nfeat, efeat, mask, label])
        
        if nfeat.isnan().any():
            logging.warning(f'nfeat has {nfeat.isnan().sum()}/{nfeat.numel()} NaNs.')
            nfeat = torch.nan_to_num(nfeat, nan=0.0)
        if efeat.isnan().any():
            logging.warning(f'efeat has {efeat.isnan().sum()}/{efeat.numel()} NaNs.')
            efeat = torch.nan_to_num(efeat, nan=0.0)

        # Train.
        optimizer.zero_grad()
        efeat_rnn = rnn.pack_sequence(torch.split(efeat, sections), enforce_sorted=False)
        emb, logits = model.forward(nfeat, efeat_rnn)
        emb = emb.detach().cpu()

        # Loss.
        embs.append(emb)
        srcs.append(src)
        loss = loss_fn(logits[train_mask], label[train_mask])

        if not loss.isnan().any():
            loss.backward()
            optimizer.step()
            
        if len(src) != len(dst):
            dst = None
        metrics.update(src, dst, emb, loss.item(), label.cpu(), logits[:, 1].detach().cpu(), mask.cpu())

    # Sort.
    src, emb = torch.cat(srcs, dim=0), torch.cat(embs, dim=0)
    sort_idx = torch.argsort(src)
    src, emb = src[sort_idx], emb[sort_idx]
    return src, emb

def reashape_input(inputs: Tensor, target: Tensor):
    if len(inputs.shape) == 2 and inputs.shape[1] == 2:
        inputs = inputs[:, 1].clip(1e-6, 1 - 1e-6)
    return inputs, target.float()


def get_loss_fn(lossfn: str, args: Args) ->  Callable[[Tensor, Tensor], Tensor]:
    """Get the loss function."""
    if lossfn == 'mse':
        def Mse(inputs: Tensor, target: Tensor):
            if len(inputs.shape) == 2 and inputs.shape[1] == 2:
                inputs = inputs[:, 1]
            return nn.MSELoss()(inputs, target.float())
        return Mse
    elif lossfn == 'ce':
        assert args.loss_weight < 1.0, 'CrossEntropy loss weight must be less than 1.0'
        loss_weight = torch.Tensor([1 - args.loss_weight, args.loss_weight]).to(args.device)
        return nn.CrossEntropyLoss(loss_weight)
    elif lossfn == 'focal':
        def Focal(inputs: Tensor, target: Tensor):
            if len(inputs.shape) == 2 and inputs.shape[1] == 2:
                inputs = inputs[:, 1]
            return torchvision.ops.sigmoid_focal_loss(inputs, target.float(), alpha=args.alpha, gamma=args.gamma, reduction='mean')
        return Focal
    else:
        raise ValueError(f'Invalid loss function: {lossfn}')

def early_stop(metrics: MetricsAll, best_metrics: MetricsAll, not_improved: float):
    if metrics > best_metrics: 
        best_metrics = metrics
        not_improved = 0.0
        # if not notqdm:
        #     print(str(metrics) + '★★★BEST VAL F1★★★')
    else:
        not_improved += 0.5
        # if not notqdm:
        #     print(metrics)
    return best_metrics, not_improved

def prepare_train(ndim: int, edim1: int, edim2: int, lossfn: str, args: Args):
    # Get Model.
    mlp_layers = list(args.mlp_layers) + [2]
    model1 = model.Encoder(ndim, edim1, args.emb_dim, mlp_layers, args)
    model2 = model.Encoder(ndim, edim2, args.emb_dim, mlp_layers, args)
    loss_fn = get_loss_fn(lossfn, args)

    # Get Optimizer.
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

    # Early Stopping.
    best_metrics1 = MetricsAll('')
    best_metrics2 = MetricsAll('')
    not_improved = 0.0

    # Trian mode.
    model1.train()
    model2.train()

    return model1, model2, loss_fn, optimizer1, optimizer2, best_metrics1, best_metrics2, not_improved

def mid_init(epoch: int, args: Args, model1, model2):
    if epoch == args.midinit:
        model1.reset_parameters()
        model2.reset_parameters()
        not_improved = 0
        best_metrics1 = MetricsAll('')
        best_metrics2 = MetricsAll('')
        return not_improved, best_metrics1, best_metrics2
    else:
        return None

def train(dataset: data.Dataset, args: Args, notqdm: bool = False, verbose: bool = True):
    """Train."""

    # Prepare train.
    start_time = time.time()
    ndim, edim1, edim2 = dataset.ndim, dataset.edim1, dataset.edim2
    model1, model2, loss_fn, optimizer1, optimizer2, best_metrics1, best_metrics2, not_improved = \
        prepare_train(ndim, edim1, edim2, args.lossfn1, args)
    
    # Function to set pbar desc.
    def set_desc(pbar, text):
        if isinstance(pbar, tqdm):
            pbar.set_description_str(text, refresh=True)

    # Function to set pbar post.
    def set_post(pbar, k1, m1, k2, m2):
        if isinstance(pbar, tqdm):
            f = lambda a: f'{a * 100:.2f}'
            auc_str = lambda k, m: f'{k}: {f(m.train.auc)}/{f(m.val.auc)}/{f(m.test.auc)}'
            pbar.set_postfix_str(f'{auc_str(k1, m1)}, {auc_str(k2, m2)}', refresh=True)

    # 1. Train.
    pbar = tqdm(range(args.epochs), total=args.epochs) if verbose else range(args.epochs)
    for epoch in pbar:
        # Init metrics.
        metrics1 = MetricsAll('M1')
        metrics2 = MetricsAll('M2')

        # Init model.
        try:
            model1.to(args.device)
            model2.to(args.device)
        except Exception as e:
            model1.cuda()
            model2.cuda()

        # Train first-order encoder.
        set_desc(pbar, f'[Train] M1')
        _, emb1 = train_encoder(model1, dataset.batchs1(), loss_fn, metrics1, optimizer1, args)
        logging.debug(f'{args.dataset} - M1 - {metrics1}')
        
        # Train second-order encoder.
        set_desc(pbar, f'[Train] M2')
        _, emb2 = train_encoder(model2, dataset.batchs2(), loss_fn, metrics2, optimizer2, args)
        logging.debug(f'{args.dataset} - M2 - {metrics2}')

        # Refresh metrics and update pbar.
        metrics1.refresh()
        metrics2.refresh()
        set_post(pbar, 'auc1', metrics1, 'auc2', metrics2)

        # Updata nfeat.
        if args.update_feat:
            emb2 = torch.cat([emb2, torch.zeros((1, emb2.shape[1])).float().to(emb2.device)], dim=0)
            emb = torch.cat([emb1, emb2[dataset.src2c_src1c]], dim=1)
            dataset.update_feat(emb)

        # Early stopping.
        best_metrics1, not_improved = early_stop(metrics1, best_metrics1, not_improved)
        best_metrics2, not_improved = early_stop(metrics2, best_metrics2, not_improved)
        if not_improved >= args.tolerance and epoch > args.midinit:
            if isinstance(pbar, tqdm):
                pbar.set_description(f'[Train]      ')
                pbar.close()
            not_improved = 0
            break

        # Mid init.
        res = mid_init(epoch, args, model1, model2)
        if res is not None:
            not_improved, best_metrics1, best_metrics2 = res

    # Updata pbar.
    if isinstance(pbar, tqdm):
        pbar.set_description(f'[Train]      ')
        pbar.close()

    # 2. Sample pair.
    batchs3, batchs4, numbatch3, numbatch4, ndim3, edim3, edim4 = dataset.pair_data(best_metrics1, best_metrics2, verbose)
    model3, model4, loss_fn, optimizer3, optimizer4, best_metrics3, best_metrics4, not_improved = \
        prepare_train(ndim3, edim3, edim4, args.lossfn3, args)

    # 3. Train pair.
    pbar = tqdm(range(args.epochs), total=args.epochs) if verbose else range(args.epochs)
    for epoch in pbar:
        # Init metrics.
        metrics3 = MetricsAll('M3')
        metrics4 = MetricsAll('M4')

        # Init model.
        try:
            model3.to(args.device)
            model4.to(args.device)
        except Exception as e:
            model3.cuda()
            model4.cuda()

        # Train first-order.
        set_desc(pbar, f'[TrainPair] M3')
        _, emb3 = train_encoder(model3, batchs3, loss_fn, metrics3, optimizer3, args)
        logging.debug(f'{args.dataset} - M3 - {metrics3}')
        
        # Train second-order.
        set_desc(pbar, f'[TrainPair] M4')
        _, emb4 = train_encoder(model4, batchs4, loss_fn, metrics4, optimizer4, args)
        logging.debug(f'{args.dataset} - M4 - {metrics4}')

        # Refresh metrics and update pbar.
        metrics3.refresh()
        metrics4.refresh()
        set_post(pbar, 'auc3', metrics3, 'auc4', metrics4)

        # Early stopping.
        best_metrics3, not_improved = early_stop(metrics3, best_metrics3, not_improved)
        best_metrics4, not_improved = early_stop(metrics4, best_metrics4, not_improved)
        if not_improved >= args.tolerance:
            if isinstance(pbar, tqdm):
                pbar.set_description(f'[TrainPair]  ')
                pbar.close()
            not_improved = 0
            break

    # Updata pbar.
    if isinstance(pbar, tqdm):
        pbar.set_description(f'[TrainPair]  ')
        pbar.close()

    # Merge metrics.
    auc, f1 = merge_metrics(best_metrics1, best_metrics2, best_metrics3, best_metrics4, verbose)

    return auc, f1

def read_and_train(args: Args):
    """Read and train a dataset."""
    # Read and train.
    util.set_seed(args.seed)
    dataset = data.get_dataset(args)
    auc, f1 = train(dataset, args)
    return auc, f1

if __name__ == '__main__':
    args = get_args('sichuan')
    auc, f1 = read_and_train(args)
    print(f'auc: {auc:.4f}, f1: {f1:.4f}')
