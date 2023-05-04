from collections import defaultdict
from functools import partial, reduce
from math import ceil
import gc
import os
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.nn.utils import rnn

from config import Args
from util import update_pbar
from metrics import MetricsAll, align_metrics
from model import find_idx
from util import norm

real_dir = os.path.dirname(__file__)

OPT_DF = Union[pd.DataFrame, None]
DATA_LIST = List[Tuple[int, List[Tuple[int, float]]]]
OPT_TS =  Optional[Tensor]

def shuffled(x: list):
    """Shuffle a list."""
    x = list(x)
    random.shuffle(x)
    return x

def encode_times(etime: List[Tensor], dim_emb: int):
    """Encode the times of each student."""
    etime_pad = rnn.pad_sequence(etime, batch_first=True)
    etime_sections = [len(t) for t in etime]
    a = 100 / torch.max(etime_pad, dim=1)[0]
    b = 1000 ** (torch.arange(0, etime_pad.shape[1] * 2, 2).to(etime_pad.device) / dim_emb)

    etime_pad = torch.Tensor((etime_pad.T * a).T * b)
    etime_pad = torch.sin(etime_pad) + torch.cos(etime_pad)
    etime_pad = torch.nan_to_num(etime_pad, nan=0.0, posinf=0.0, neginf=0.0)
    etime = [line[:section] for line, section in zip(etime_pad, etime_sections)]
    return etime

def stat_missing(x: Tensor):
    a = int((x == -1).sum())
    b = int(x.numel())
    return f'{a}/{b}({a / b * 100:.2f}%)'

def split_dataset(pos_ids: Set[int], neg_ids: Set[int], split_ratio: List[float], balance: bool):
    """Splits the dataset into train, val and test sets."""
    assert sum(split_ratio) == 1.0
    if balance:
        min_length = min(len(pos_ids), len(neg_ids))
        pos_ids = set(shuffled(list(pos_ids))[:min_length])
        neg_ids = set(shuffled(list(neg_ids))[:min_length])
    ids = torch.Tensor(shuffled(list(pos_ids | neg_ids))).long()
    sections = [int(len(ids) * s) for s in split_ratio]
    sections[-1] = len(ids) - sum(sections[:-1])
    tensors = torch.split(ids, sections)
    return [{int(x) for x in tensor} for tensor in tensors]

def get_batch_idx(sections: List[int], batch_size: int):
    """Get the mask of batchs."""
    idx = torch.randperm(len(sections)).long().tolist()  # Shuffle nodes.
    batch_dict: Dict[int, int] = defaultdict(int)  # Batch of each node.

    num_data = 0
    batch_id = 0
    node_in_batch = 0  # Ensure that each batchsize has at least two nodes.
    for x in idx:
        section = sections[x]
        num_data += section
        batch_dict[x] = batch_id
        node_in_batch += 1
        if num_data > batch_size and node_in_batch >= 2:
            batch_id += 1
            num_data = 0
            node_in_batch = 0
        
    mask = torch.Tensor([batch_dict[i] for i, section in enumerate(sections) for _ in range(section)]).long()
    num_batch = int(mask.max()) + 1
    return mask, num_batch

def get_train_mask(src: Tensor, train_ids: Set[int], val_ids: Set[int], test_ids: Set[int], nounq: bool):
    def mask(x: int):
        if x in train_ids:
            return 0
        elif x in val_ids:
            return 1
        elif x in test_ids:
            return 2
        else:
            return -1
    if not nounq:
        src = torch.unique(src)
    return torch.Tensor([mask(int(x)) for x in src]).long()

def get_sections(edge_df: pd.DataFrame):
    """Get the number of edges of each student."""
    srcs = torch.from_numpy(edge_df['src'].to_numpy()).long()
    _, cnt = torch.unique(srcs, return_counts=True)
    sections = torch.Tensor(cnt).long().tolist()
    return sections
    
def get_pair_efeat(pairid: Tensor, src: Tensor, efeat: Tensor, etime: Tensor, sample: int):
    """Get the edge features of each pair.
    Returns:
        efeat_list: List[Tensor], List of edge features of each pair, length = num_pairs.
        etime_list: List[Tensor], List of edge times of each pair, length = num_pairs.
        idx_efeat_pairid: Tensor, Index of each pair in pairid, length = num_pairs.
        section: List[int], Number of edges of each pair, length = num_pairs.
    """
    unq, cnt = torch.unique(src, return_counts=True)
    efeat_list = torch.split(efeat, cnt.tolist())
    etime_list = torch.split(etime, cnt.tolist())

    sample = int(sample / 2)
    efeat_list = [x[torch.argsort(y)[:sample]] for x, y in zip(efeat_list, etime_list)]
    etime_list = [x[torch.argsort(y)[:sample]] for x, y in zip(etime_list, etime_list)]

    idx_efeat_pairid = find_idx(pairid, unq)
    sections = [len(efeat_list[i]) for i in idx_efeat_pairid]
    return efeat_list, etime_list, idx_efeat_pairid, sections

def merge_pair_efeat(src_efeat_list: List[Tensor], src_etime_list: List[Tensor], 
                        dst_efeat_list: List[Tensor], dst_etime_list: List[Tensor],
                        sample: int):
    efeat_list = [torch.cat([x, y], dim=0) for x, y in zip(src_efeat_list, dst_efeat_list)]
    idx_list = [torch.argsort(torch.cat([x, y], dim=0))[:sample] for x, y in zip(src_etime_list, dst_etime_list)]
    efeat_list = [x[argsort] for x, argsort in zip(efeat_list, idx_list)]
    sections = [len(x) for x in efeat_list]
    efeat = torch.cat(efeat_list, dim=0)
    return efeat, sections

class Dataset:
    """Dataset for Cobifraud."""
    def __init__(self,
        node_df: pd.DataFrame, 
        nfeat1: Tensor, 
        edge1_df: pd.DataFrame, 
        efeat1: Tensor, 
        etime1: Tensor, 
        edge2_df: pd.DataFrame, 
        efeat2: Tensor, 
        etime2: Tensor,
        pair_df: pd.DataFrame,
        args: Args,
        notqdm: bool = False
        ):

        # Init pbar.
        if notqdm:
            pbar = None
        else:
            pbar = tqdm(total=9, desc='[CobiDataset] Create dataset')

        # 1. Init attrs.
        self.node_df = node_df
        self.nfeat1 = nfeat1
        self.edge1_df = edge1_df
        self.efeat1 = efeat1
        self.etime1 = etime1
        self.edge2_df = edge2_df
        self.efeat2 = efeat2
        self.etime2 = etime2
        self.pair_df = pair_df
        self.args = args
        update_pbar(pbar)

        # 2. Get nid.
        self.nid = torch.Tensor(node_df['id'].to_numpy()).long()
        update_pbar(pbar)

        # 3. Get pos_ids and neg_ids.
        pos_ids = set(node_df[node_df['label'] == 1]['id'])
        neg_ids = set(node_df[node_df['label'] == 0]['id'])
        self.pos_ids, self.neg_ids = pos_ids, neg_ids
        update_pbar(pbar)

        # 4. Split the dataset.
        self.train_ids, self.val_ids, self.test_ids = split_dataset(pos_ids, neg_ids, args.split_ratio, args.balance_split)
        update_pbar(pbar)

        # 5. Get batchs.
        sections1, sections2 = get_sections(edge1_df), get_sections(edge2_df)
        self.sections1, self.sections2 = sections1, sections2
        self.batch_mask1, self.numbatch1 = get_batch_idx(sections1, args.batch_size)
        self.batch_mask2, self.numbatch2 = get_batch_idx(sections2, args.batch_size)
        update_pbar(pbar)

        # 6. Get mapping.
        func = lambda df, key: torch.from_numpy(df[key].to_numpy()).long()
        func1 = lambda key: func(edge1_df, key)
        func2 = lambda key: func(edge2_df, key)
        src1, dst1, src2, dst2 = func1('src'), func1('dst'), func2('src'), func2('dst')
        self.src1, self.dst1, self.src2, self.dst2 = src1, dst1, src2, dst2
        self.src1c_nid = find_idx(self.nid, torch.unique(src1))
        self.src1c_dst1 = find_idx(dst1, torch.unique(src1))
        self.src1c_dst2 = find_idx(dst2, torch.unique(src1))
        self.src2c_src1c = find_idx(torch.unique(src1), torch.unique(src2))
        update_pbar(pbar)

        # 7. Cat efeat.
        nfeat1, dstfeat1, dstfeat2 = self.generate_emb()
        self.nfeat1 = torch.cat([self.nfeat1, nfeat1], dim=1)
        self.efeat1 = torch.cat([self.efeat1, self.etime1, dstfeat1], dim=1)
        self.efeat2 = torch.cat([self.efeat2, self.etime2, dstfeat2], dim=1)
        update_pbar(pbar)

        # 8. Get the final length of the efeat.
        self.ndim = int(self.nfeat1.shape[1])
        self.edim1 = int(self.efeat1.shape[1])
        self.edim2 = int(self.efeat2.shape[1])
        update_pbar(pbar)

        # 9. Logging info.
        name_str = str(args.dataset).capitalize()
        num_nodes, num_feat1, num_feat2, num_featp = len(self.nid), len(self.efeat1), len(self.efeat2), len(self.pair_df)
        nodes_str = f'nodes={num_nodes:,}'
        edges_str = f'edges={num_feat1}({num_feat1/num_nodes:.1f}X)-{num_feat2}({num_feat2/num_nodes:.1f}X)-{num_featp}({num_featp/num_nodes:.1f}X)'
        num_pos, num_neg = len(self.pos_ids), len(self.neg_ids)
        pos_str = f'pos={num_pos:,}({num_pos / (num_pos + num_neg):.2%})'
        logging.info(f'{name_str}: {nodes_str} {edges_str} {pos_str}')
        update_pbar(pbar, True)
       

    def generate_emb(self):
        func = torch.randn if self.args.embmode == 'randn' else torch.zeros
        nfeat1 = func((len(self.nfeat1), self.args.emb_dim * 2), dtype=torch.float32)
        dstfeat1 = func((len(self.dst1), self.args.emb_dim * 2), dtype=torch.float32)
        dstfeat2 = func((len(self.dst2), self.args.emb_dim * 2), dtype=torch.float32)
        return nfeat1, dstfeat1, dstfeat2

    def update_feat(self, emb1: Tensor):
        """Update the efeat."""
        assert len(emb1) == len(self.sections1), f'{len(emb1)} != {len(self.sections1)}'
        assert emb1.shape[1] == self.args.emb_dim * 2, f'{emb1.shape[1]} != {self.args.emb_dim * 2}'
        self.nfeat1[:, -self.args.emb_dim * 2:] = emb1[self.src1c_nid]
        self.efeat1[:, -self.args.emb_dim * 2:] = emb1[self.src1c_dst1]
        self.efeat2[:, -self.args.emb_dim * 2:] = emb1[self.src1c_dst2]

    def get_label(self, src: Tensor):
        """Get node label."""
        def label(x: int):
            if x in self.pos_ids:
                return 1
            elif x in self.neg_ids:
                return 0
            else:
                return -1
        return torch.Tensor([label(int(x)) for x in torch.unique(src)]).long()

    def get_batch(self, i: int, batch_mask: Tensor, src: Tensor, dst: Tensor, nfeat: Tensor, efeat: Tensor):
        batch_mask = batch_mask == i
        src = src[batch_mask]
        dst = dst[batch_mask]
        efeat = efeat[batch_mask]

        sections: List[int] = torch.unique(src, return_counts=True)[1].tolist()
        assert sum(sections) == len(efeat), f'{sum(sections)} != {len(efeat)}'
        label = self.get_label(src)
        batch_mask = get_train_mask(src, self.train_ids, self.val_ids, self.test_ids, False)
        src = torch.unique(src)

        nfeat = torch.cat([nfeat, torch.randn([1, nfeat.shape[1]])], dim=0)
        nfeat_idx = find_idx(src, self.nid)
        nfeat = nfeat[nfeat_idx]
        return src, dst, nfeat, efeat, sections, batch_mask, label
        
    def batchs1(self):
        """Get batches of graph1."""
        return [self.get_batch(i, self.batch_mask1, self.src1, self.dst1, self.nfeat1, self.efeat1) for i in range(self.numbatch1)]

    def batchs2(self):
        """Get batches of graph2."""
        return [self.get_batch(i, self.batch_mask2, self.src2, self.dst2, self.nfeat1, self.efeat2) for i in range(self.numbatch2)]

    def pair_data(self, metric1: MetricsAll, metric2: MetricsAll, verbose: bool):
        """Get batches of pairs."""
        pbar = tqdm(total = 10, desc = '[SamplePair] ', disable=not verbose)

        # 1. Align the metrics.
        pair_df = self.pair_df.dropna(how='any')
        nid, label, mask, emb1, emb2, pred1, pred2 = align_metrics(metric1, metric2)
        pred_label1, pred_label2 = (pred1 > 0.5).long(), (pred2 > 0.5).long()
        pbar.update()

        # 2. Merge the metrics.
        pred_df = pd.DataFrame({'src': nid, 'src_pred1': pred1, 'src_pred2': pred2, 'src_mask': mask.long()})
        pair_df = pair_df.merge(pred_df, on='src', how='left')
        pred_df = pred_df.rename(columns={'src': 'dst', 'src_pred1': 'dst_pred1', 'src_pred2': 'dst_pred2', 'src_mask': 'dst_mask'})
        pair_df = pair_df.merge(pred_df, on='dst', how='left').fillna(0.5)
        pbar.update()

        # 3. Get pair label.
        assert pair_df is not None
        to_tensor = lambda x: torch.from_numpy(x.to_numpy())
        pair_df['pair_label'] = (pair_df['src_label'] + pair_df['dst_label']).astype(int)
        pair_df['pair_pred_label'] = (pair_df['src_pred1'] + pair_df['dst_pred1']).round(0).astype(int)
        pbar.update()

        # 4. Sample pairs.
        pair_df = sample_pair(pair_df, self.edge1_df, self.edge2_df, self.args)
        pbar.update()

        # 5. Get mask.
        def func(x, y):
            if x == 0 and y == 0:
                return 0
            elif x + y  == 3:
                return -1
            elif x ** 2 + y ** 2 <= 2:
                return 1
            else:
                return 2
        pair_df['pair_train_mask'] = pair_df.apply(lambda x: func(x['src_mask'], x['dst_mask']), axis=1)
        pair_df = pair_df.drop(columns=['src_mask', 'dst_mask'])
        pbar.update()
        
        # 6. Get pair nfeat.
        pair_src, pair_dst = to_tensor(pair_df['src']).long(), to_tensor(pair_df['dst']).long()
        pair_label, pair_train_mask = to_tensor(pair_df['pair_label']).long(), to_tensor(pair_df['pair_train_mask']).long()
        pair_label = (pair_label == 2).long()
        pred_ts = to_tensor(pair_df[['src_pred1', 'src_pred2', 'dst_pred1', 'dst_pred2']])
        idx_emb_src = find_idx(pair_src, nid)
        idx_emb_dst = find_idx(pair_dst, nid)
        pair_nfeat = torch.cat([pred_ts, emb1[idx_emb_src], emb2[idx_emb_src], emb1[idx_emb_dst], emb2[idx_emb_dst]], dim=1)
        nid = to_tensor(self.node_df['id']).long()
        pair_nfeat = torch.cat([pair_nfeat, self.nfeat1[find_idx(pair_src, nid)], self.nfeat1[find_idx(pair_dst, nid)]], dim=1).nan_to_num(0)
        assert not pair_nfeat.isnan().any(), f'pair_nfeat has nan: {pair_nfeat.isnan().sum()}/{pair_nfeat.shape[0] * pair_nfeat.shape[1]}\n{pair_nfeat}'
        ndim = pair_nfeat.shape[1]
        pbar.update()

        # 7. Get pair efeat.
        pair_efeat: List[Tuple[List[Tensor], List[Tensor], Tensor, list[int]]] = []  # od1-src, od1-dst, od2-src, od2-dst
        etime1, etime2 = to_tensor(self.edge1_df['time']), to_tensor(self.edge2_df['time'])
        for src, efeat, etime, sample in [(self.src1, self.efeat1, etime1, self.args.sample3), (self.src2, self.efeat2, etime2, self.args.sample4)]:
            for pairid in [pair_src, pair_dst]:
                efeat_list, etime_list, idx_efeat_pairid, sections = get_pair_efeat(pairid, src, efeat, etime, sample)
                assert len(sections) == len(pairid), f'{len(sections)} != {len(pairid)}'
                pair_efeat.append((efeat_list, etime_list, idx_efeat_pairid, sections))
        edim1, edim2 = pair_efeat[0][0][0].shape[1], pair_efeat[2][0][0].shape[1]
        pbar.update()

        # 8. Merge sections.
        pair_sections1 = [x + y for x, y in zip(pair_efeat[0][3], pair_efeat[1][3])]
        pair_sections2 = [x + y for x, y in zip(pair_efeat[2][3], pair_efeat[3][3])]
        pbar.update()

        # 9. Split batch size
        pair_batch_mask1, pair_numbatch1 = get_batch_idx(pair_sections1, self.args.batch_size)
        pair_batch_mask2, pair_numbatch2 = get_batch_idx(pair_sections2, self.args.batch_size)
        pbar.update()

        # 10. Get pair batchs.
        batchs1 = [x for x in get_pair_batchs(pair_batch_mask1, pair_numbatch1, pair_src, pair_dst, pair_sections1, pair_nfeat, pair_efeat[:2], pair_label, pair_train_mask, self.args.sample3)]
        batchs2 = [x for x in get_pair_batchs(pair_batch_mask2, pair_numbatch2, pair_src, pair_dst, pair_sections2, pair_nfeat, pair_efeat[2:], pair_label, pair_train_mask, self.args.sample4)]
        pbar.update()

        return batchs1, batchs2, pair_numbatch1, pair_numbatch2, ndim, edim1, edim2

def sample_pair(pair_df: pd.DataFrame, edge1_df: pd.DataFrame, edge2_df: pd.DataFrame, args: Args):
    target = int(len(edge1_df) / args.sample3)

    pair_df = pair_df[pair_df['src_mask'] + pair_df['dst_mask'] != 3]
    pred_label = pair_df['pair_pred_label'].to_numpy().astype(np.int64)
    
    def get_idx(pred_label: np.ndarray, target: int):
        idxs = []
        for label_value in [0, 1, 2]:
            idx = np.where(pred_label == label_value)[0]
            np.random.shuffle(idx)
            idx = idx[:target]
            idxs.append(idx)
        nn_idx, pn_idx, pp_idx = idxs
        return nn_idx, pn_idx, pp_idx

    nn_idx, pn_idx, pp_idx = get_idx(pred_label, target)
    idx = np.concatenate([nn_idx, pn_idx, pp_idx])
    pair_df = pair_df.iloc[idx]

    keys = ['pair_label', 'pair_pred_label']
    stat_df = pair_df[keys].groupby(keys).size().reset_index(name='count')
    stat_df['count'] = stat_df['count'] / len(pair_df)
    stat_df = stat_df.sort_values(by=keys)

    ids = np.unique(np.concatenate([pair_df['src'].to_numpy(), pair_df['dst'].to_numpy()]))
    func1 = lambda x: len(np.where(pair_df['pair_label'].to_numpy() == x)[0])
    func2 = lambda x: len(np.where(pair_df['pair_pred_label'].to_numpy() == x)[0])
    logging.debug(f'num of ids in pair_df: {len(ids)}, pp={func1(2)}/{func2(2)}, pn={func1(1)}/{func2(1)}, nn={func1(0)}/{func2(0)}')
    return pair_df

def merge_efeat(node_batch_mask: Tensor, efeat: List[Tuple[List[Tensor], List[Tensor], Tensor, List[int]]], sample: int):
    src_efeat_list, src_etime_list, src_idx_efeat_pairid, src_sections = efeat[0]
    dst_efeat_list, dst_etime_list, dst_idx_efeat_pairid, dst_sections = efeat[1]
    assert len(node_batch_mask) == len(src_idx_efeat_pairid) == len(dst_idx_efeat_pairid), f'{len(node_batch_mask)} != {len(src_idx_efeat_pairid)} != {len(dst_idx_efeat_pairid)}'

    src_idx_efeat_pairid = src_idx_efeat_pairid[node_batch_mask]
    dst_idx_efeat_pairid = dst_idx_efeat_pairid[node_batch_mask]

    src_efeat_list = [src_efeat_list[i] for i in src_idx_efeat_pairid]
    src_etime_list = [src_etime_list[i] for i in src_idx_efeat_pairid]
    dst_efeat_list = [dst_efeat_list[i] for i in dst_idx_efeat_pairid]
    dst_etime_list = [dst_etime_list[i] for i in dst_idx_efeat_pairid]
    efeat_list = [torch.cat([x, y], dim=0) for x, y in zip(src_efeat_list, dst_efeat_list)]
    etime_list = [torch.cat([x, y], dim=0) for x, y in zip(src_etime_list, dst_etime_list)]
    
    arg_sort = [torch.argsort(x)[:sample] for x in etime_list]
    assert len(arg_sort) == len(efeat_list), f'{len(arg_sort)} != {len(efeat_list)}'
    efeat_list = [x[y] for x, y in zip(efeat_list, arg_sort)]
    sections = [len(x) for x in efeat_list]

    return efeat_list, etime_list, sections


def get_pair_batchs(batch_mask: Tensor, numbatch: int, src: Tensor, dst: Tensor, sections: List[int],
                    nfeat: Tensor, efeat: List[Tuple[List[Tensor], List[Tensor], Tensor, List[int]]], 
                    label: Tensor, train_mask: Tensor, sample: int):
    assert len(sections) == len(src) == len(dst) == len(nfeat) == len(label) == len(train_mask), f'{len(sections)} != {len(src)} != {len(dst)} != {len(nfeat)} != {len(label)} != {len(train_mask)}'
    cum_sum = np.cumsum(sections)
    node_idx = torch.from_numpy(np.concatenate([[0], cum_sum[:-1]])).long()
    node_batch_mask = batch_mask[node_idx]

    for i in range(numbatch):
        _node_batch_mask = node_batch_mask == i

        batch_efeat_list, batch_etime_list, batch_sections = merge_efeat(_node_batch_mask, efeat, sample)
        batch_efeat = torch.cat(batch_efeat_list, dim=0)

        batch_src = src[_node_batch_mask]
        batch_dst = dst[_node_batch_mask]
        batch_nfeat = nfeat[_node_batch_mask]
        batch_label = label[_node_batch_mask]
        batch_train_mask = train_mask[_node_batch_mask]

        batch_nfeat = norm(batch_nfeat.nan_to_num(0))
        batch_efeat = norm(batch_efeat.nan_to_num(0))
        yield batch_src, batch_dst, batch_nfeat, batch_efeat, batch_sections, batch_train_mask, batch_label