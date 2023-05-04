from __future__ import annotations

import functools
import os
from typing import List, Optional, Tuple
import xgboost as xgb

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import Tensor

from model import find_idx

real_dir = os.path.dirname(__file__)

class Metrics:
    def __init__(self, name: str):
        self.name = name
        self.metrics = ['acc', 'auc', 'f1', 'macro-f1', 'prec', 'recall']
        self.src: List[Tensor] = []
        self.dst: List[Optional[Tensor]] = []
        self.emb: List[Tensor] = []
        self.label: List[Tensor] = []
        self.pred: List[Tensor] = []

        self.acc: float = -1.0
        self.auc: float = -1.0
        self.f1: float = -1.0
        self.marco_f1: float = -1.0
        self.prec: float = -1.0
        self.recall: float = -1.0

    def get_metrics(self):
        if len(self.label) == 0 or len(self.pred) == 0:
            # logging.warning('No label or pred found.')
            self.acc = -1
            self.auc = -1
            self.f1 = -1
            self.prec = -1
            self.recall = -1
            return [0.0] * len(self.metrics)
            
        label = np.array(torch.cat(self.label, dim=0).numpy()).astype(np.int64)
        pred = np.array(torch.cat(self.pred, dim=0).numpy()).astype(np.float32)
        pred_label = np.array(pred > 0.5).astype(np.int64)

        marco_f1_func = functools.partial(f1_score, average='macro')

        # print(f'{self.name:<5s}', 'pred ', (pred_label==1).sum(), pred_label.shape, pred_label[:10].tolist())
        # print(f'{self.name:<5s}', 'label', (label==1).sum(), label.shape, label[:10].tolist())

        self.acc = float(accuracy_score(label, pred_label))
        self.auc = float(roc_auc_score(label, pred))
        func = lambda f: float(f(label, pred_label, zero_division=0))
        self.f1, self.marco_f1, self.prec, self.recall = map(func, [f1_score, marco_f1_func, precision_score, recall_score])

        return [self.acc, self.auc, self.f1, self.marco_f1, self.prec, self.recall]

    def get_pred(self):
        src = torch.cat(self.src, dim=0).long()
        dst: Optional[Tensor]
        if self.dst[0] is not None:
            dst = torch.cat(self.dst, dim=0).long()  # type: ignore
            assert len(src) == len(dst), f'{len(src)} != {len(dst)}'  # type: ignore
        else:
            dst = None
        emb = torch.cat(self.emb, dim=0)
        label = torch.cat(self.label, dim=0).long()
        pred = torch.cat(self.pred, dim=0).float()
        return src, dst, emb, label, pred


    def update(self, src: Tensor, dst: Optional[Tensor], emb: Tensor, label: Tensor, pred: Tensor):
        assert label.shape == pred.shape, f'Label and pred shape mismatch: {label.shape} != {pred.shape}'
        assert len(src) == len(emb) == len(label) == len(pred), f'{len(src)}, {len(emb)}, {len(label)}, {len(pred)}'
        idx = ~torch.isnan(pred)
        src, emb, label, pred = src[idx], emb[idx], label[idx], pred[idx]
        assert len(src) == len(label) == len(pred), f'{len(src)}, {len(label)}, {len(pred)}'
        self.src.append(src.detach().cpu())

        if dst is not None:
            dst = dst[idx]
            self.dst.append(dst.detach().cpu())
        else:
            self.dst.append(None)

        self.emb.append(emb.detach().cpu())
        self.label.append(label.detach().cpu())
        self.pred.append(pred.detach().cpu())


class MetricsAll():
    """Metrics for train, val, and test."""

    def __init__(self, name: str):
        self.name = name
        self.loss: List[float] = []
        self.train = Metrics('train')
        self.val = Metrics('val')
        self.test = Metrics('test')

    def update(self, src: Tensor, dst: Optional[Tensor], emb: Tensor, loss: float, label: Tensor, pred: Tensor, mask: Tensor):
        assert len(src) == len(emb) == len(label) == len(pred) == len(mask), f'{len(src)} != {len(emb)} != {len(label)} != {len(pred)} != {len(mask)}'
        assert src.shape == label.shape == pred.shape == mask.shape, f'{src.shape} != {label.shape} != {pred.shape} != {mask.shape}'
        loss = float(loss)

        if str(loss) != 'nan':
            self.loss.append(loss)

        def update_metrics(metrics: Metrics, mask: Tensor):
            if dst is not None:
                assert (len(src) == len(dst)), f'{len(src)} != {len(dst)}'
                _dst = dst[mask]
            else:
                _dst = None
            metrics.update(src[mask], _dst, emb[mask], label[mask], pred[mask])

        update_metrics(self.train, mask == 0)
        update_metrics(self.val, mask == 1)
        update_metrics(self.test, mask == 2)

    def __str__(self):
        res = f'{self.name} loss:{np.mean(self.loss):.5f}'
        zipped = zip(self.train.metrics, self.train.get_metrics(),
                     self.val.get_metrics(), self.test.get_metrics())
        for name, train_metric, val_metric, test_metric in zipped:
            if name in ['auc', 'macro-f1']:
                res += f' {name}:{train_metric:.4f}/{val_metric:.4f}/{test_metric:.4f}'
        return res

    def get_label(self) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
        train_src, train_dst, train_emb, train_label, train_pred = self.train.get_pred()
        val_src, val_dst, val_emb, val_label, val_pred = self.val.get_pred()
        test_src, test_dst, test_emb, test_label, test_pred = self.test.get_pred()
        
        src = torch.cat([train_src, val_src, test_src], dim=0)
        dst = torch.cat([train_dst, val_dst, test_dst], dim=0) if train_dst is not None else None  # type: ignore
        emb = torch.cat([train_emb, val_emb, test_emb], dim=0)
        label = torch.cat([train_label, val_label, test_label], dim=0)
        pred = torch.cat([train_pred, val_pred, test_pred], dim=0)
        mask = torch.cat([torch.zeros_like(train_src), torch.ones_like(val_src), torch.ones_like(test_src) * 2], dim=0).long()

        sort = torch.argsort(src)
        src, emb, label, pred, mask = map(lambda x : x[sort], [src, emb, label, pred, mask])

        return src, dst, emb, label, pred, mask
    
    def refresh(self):
        self.train.get_metrics()
        self.val.get_metrics()
        self.test.get_metrics()

    def __lt__(self, other: MetricsAll):
        return self.val.f1 < other.val.f1

    def __gt__(self, other: MetricsAll):
        return self.val.f1 > other.val.f1

    def __eq__(self, other: MetricsAll):
        return self.val.f1 == other.val.f1

def align_metrics(metrics1: MetricsAll, metrics2: MetricsAll):
    """Align two metrics by src."""
    src1, dst1, emb1, label1, pred1, mask1 = metrics1.get_label()
    src2, dst2, emb2, label2, pred2, mask2 = metrics2.get_label()
    idx = find_idx(src2, src1)
    valid = idx != -1
    src1, emb1, label1, pred1, mask1 = map(lambda x : x[idx][valid], [src1, emb1, label1, pred1, mask1])
    src2, emb2, label2, pred2, mask2 = map(lambda x : x[valid], [src2, emb2, label2, pred2, mask2])

    assert (src1==src2).sum() == len(src1), f'{(src1!=src2).sum()}/{len(src1)} not equal.'
    assert (label1==label2).sum() == len(label1), f'{(label1!=label2).sum()}/{len(label1)} not equal.'
    assert (mask1==mask2).sum() == len(mask1), f'{(mask1!=mask2).sum()}/{len(mask1)} not equal.'

    nid, label, mask = src1, label1, mask1
    return nid, label, mask, emb1, emb2, pred1, pred2

def align_pair_metrics(metrics3: MetricsAll, metrics4: MetricsAll):
    src3, dst3, emb3, label3, pred3, mask3 = metrics3.get_label()
    src4, dst4, emb4, label4, pred4, mask4 = metrics4.get_label()
    assert dst3 is not None and dst4 is not None, 'dst3 and dst4 should not be None.'

    def get_df(src: Tensor, dst: Tensor, pred: Tensor, label: Tensor):
        nid = torch.cat([src, dst], dim=0).long()
        pred = torch.cat([pred, pred], dim=0).float()
        label = torch.cat([label, label], dim=0).long()
        return pd.DataFrame({'id': nid.numpy(), 'pred': pred.numpy(), 'label': label.numpy()})

    df3 = get_df(src3, dst3, pred3, label3)
    df4 = get_df(src4, dst4, pred4, label4)

    def group_stat(df: pd.DataFrame):
        df = df.groupby('id').agg({'pred': ['mean', 'max', 'min', 'std']})
        df = df.rename(columns={'mean': 'pred_mean', 'max': 'pred_max', 'min': 'pred_min', 'std': 'pred_std'})
        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        return df

    pair_pred_df3 = group_stat(df3)
    pair_pred_df4 = group_stat(df4)
    pair_pred_df = pair_pred_df3.merge(pair_pred_df4, on='id', how='outer', suffixes=('_3', '_4')).fillna(0.0)

    return pair_pred_df

def get_metrics(labels, preds):
    auc = float(roc_auc_score(labels, preds))
    mf1 = float(f1_score(labels, preds > 0.5, average='macro'))
    return auc, mf1

def merge_metrics(metrics1: MetricsAll, metrics2: MetricsAll, metrics3: MetricsAll, metrics4: MetricsAll, verbose: bool):
    """Merge metrics1 and metrics2."""

    # Align metrics
    nid, label, mask, emb1, emb2, pred1, pred2 = align_metrics(metrics1, metrics2)
    res_df = pd.DataFrame({'id': nid.numpy(), 'label': label.numpy(), 'mask': mask.numpy(), 'pred1': pred1.numpy(), 'pred2': pred2.numpy()})
    pair_pred_df = align_pair_metrics(metrics3, metrics4)
    res_df = pd.merge(res_df, pair_pred_df, on='id', how='left').fillna(0.5)

    # Convert to tensor
    x = res_df[['pred1', 'pred2', 'pred_mean_3', 'pred_max_3', 'pred_std_3', 'pred_min_3', 'pred_mean_4', 'pred_max_4', 'pred_std_4', 'pred_min_4']].to_numpy()
    y = res_df['label'].to_numpy()
    mask = res_df['mask'].to_numpy()

    # Train by xgboost
    def xgb_train(x, y):
        dtrain = xgb.DMatrix(x[mask==0], label=y[mask==0])
        dtest = xgb.DMatrix(x[mask==2], label=y[mask==2])
        param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        bst = xgb.train(param, dtrain, 20)
        pred = bst.predict(dtest)
        auc, f1 = get_metrics(y[mask==2], pred)
        return auc, f1
    auc, f1 = xgb_train(x, y)

    return auc, f1
    