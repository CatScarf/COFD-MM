import gc
import os
import time
import logging
import sqlite3
from os.path import exists, join
from typing import List, Union

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from torch import Tensor

from util import cache, update_pbar, mem_usage, norm
from config import Args
from dataset import Dataset
from dataset import find_idx

real_dir = os.path.dirname(__file__)

OPT_DF = Union[pd.DataFrame, None]

@cache
def read_node_edge(dir: str, name: str):
    """Reads a csv file from the data_csv_norm directory."""
    def read_if_exists(suffix: str) -> OPT_DF:
        path = join(dir, f'{name}_{suffix}.csv')
        return pd.read_csv(path) if exists(path) else None
    node_df = read_if_exists('node')
    edge_df = read_if_exists('edge')
    assert edge_df is not None
    return node_df, edge_df



def to_undirect(edge_df: pd.DataFrame, efeat: Tensor):
    """Convert direct graph to undirect graph."""
    edge_df_r = edge_df.copy().rename(columns={'src': 'dst', 'dst': 'src'})
    edge_df = pd.concat([edge_df, edge_df_r], axis=0)
    efeat = torch.cat([efeat, efeat], dim=0)
    return edge_df, efeat

def encode_time(etime: Tensor, sections: List[int], emb_dim: int):
    """Encodes time to a vector.
    phi_{2j}(t_i) = sin(100/T_{max}*t_i/1000^{2i/D_{time}})
    phi_{2j+1}(t_i) = cos(100/T_{max}*t_i/1000^{2i/D_{time}})
    """
    i = torch.Tensor([i for x in sections for i in range(x)]).float()
    is_sin = torch.Tensor([True if i % 2 == 0 else False for x in sections for i in range(x)]).bool()
    t_max = torch.Tensor([x.max() for x in torch.split(etime, sections) for _ in range(len(x))]).float()

    a = (100 / t_max)[None, :]                            # (1, n)
    b = (2 * i)[None, :]                                  # (1, n)
    b = b / (torch.arange(emb_dim).float()[:, None] + 1)  # (d, n)
    b = torch.pow(10000, b)                               # (d, n)
    c = (a * etime[None, :] / b).T                        # (n, d)
    etime = torch.where(is_sin[:, None], torch.sin(c), torch.cos(c))  # (n, d)
    etime = etime.nan_to_num(0.0)
    return etime

def extract(node_df: OPT_DF, edge_df: pd.DataFrame):
    """Extract features from node_df and edge_df.
    
    Returns:
        node_df: Node dataframe, with columns ['id', 'label'].
        edge_df: Edge dataframe, with columns ['src', 'dst', 'time'].
        nfeat: Node features, with shape (num_nodes, d).
        efeat: Edge features, with shape (num_edges, d).
    """
    # Get efeat and edge_df, edge_df with columns ['src', 'dst', 'time].
    keys_feat = ['time'] + [k for k in edge_df.columns if k.startswith('feat')]
    keys_feat = ['type'] + keys_feat if type in edge_df.columns else keys_feat
    efeat = torch.Tensor(edge_df[keys_feat].to_numpy()).float()
    node_df_new = None
    if 'label' in edge_df.columns:
        assert node_df is None, str(node_df)
        pos_ids = set(edge_df[edge_df['label'] == 1]['src'].to_numpy())
        neg_ids = set(edge_df[edge_df['label'] == 0]['src'].to_numpy())
        neg_ids = neg_ids - pos_ids
        node_df_new = pd.DataFrame({'id': list(pos_ids | neg_ids), 'label': [1 if i in pos_ids else 0 for i in pos_ids | neg_ids]})

    # Get etime.
    if 'time' in edge_df.columns:
        edge_df_new = edge_df[['src', 'dst', 'time']]
    else:
        edge_df_new = edge_df[['src', 'dst']]
        edge_df_new['time'] = np.zeros((len(edge_df_new), ), dtype=np.float32)

    # Get nfeat and node_df, node_df with columns ['id'], ['label'].
    nfeat = None
    if node_df is not None:
        keys_feat = [k for k in node_df.keys() if k.startswith('feat')]
        nfeat = torch.Tensor(node_df[keys_feat].values).float()
        node_df_new = node_df[['id', 'label']]
    assert node_df_new is not None, str(node_df_new)
    if nfeat is None:
        nfeat = torch.zeros((len(node_df_new), 1)).float()

    # Astype.
    node_df_new = node_df_new.astype({'id': int, 'label': int})
    edge_df_new = edge_df_new.astype({'src': int, 'dst': int, 'time': float})

    return node_df_new, edge_df_new, nfeat, efeat


def alter_table(new_table: str, old_table: str, cursor: sqlite3.Cursor):
    """Alter new_table to old_table.
    
    Args:
        new_table: New table name.
        old_table: Old table name.
        cursor: Cursor of sqlite3.
    """
    cursor.execute(f'DROP TABLE IF EXISTS {old_table}')
    cursor.execute(f'ALTER TABLE {new_table} RENAME TO {old_table}')


def get_edge_od2(node_df: pd.DataFrame, edge_df: pd.DataFrame, mode: int, ford: bool, sord: bool, notqdm: bool):
    """Get second order edges from first order edges.

    Args:
        node_df: The dataframe of nodes, with columns ['id', ...].
        edge_df: First order edges, with columns ['src', 'dst'].
        mode: Second-order neighbor connection mode.
            0: src -> dst == dst <- src
            1: src -> dst == src -> dst
        ford: Whether to de-duplicate for first-order edges, false for this item is likely to cause a memory explosion.
        sord: Whether to de-duplicate for second-order edges.
    Returns:
        edge2_df: Second order edges, with columns ['src', 'mid', 'dst', 'idx1', 'idx2'].
    """
    if notqdm:
        pbar = None
    else:
        pbar = tqdm(total=5, desc='[get_edge_od2]', leave=False)

    # 1. Connect to database.
    time_str = f'{time.time()}'
    db_path = os.path.join(real_dir, f'{time_str}.db')
    conn = sqlite3.connect(db_path)
    edge_df = edge_df.copy()
    edge_df['idx'] = np.arange(len(edge_df))
    node_df.to_sql('node', conn, index=False, if_exists='replace')
    edge_df.to_sql('edge', conn, index=False, if_exists='replace')
    cursor = conn.cursor()
    update_pbar(pbar)

    # 2. De-duplicate 1-order edges.
    if ford:
        cursor.execute("CREATE TABLE temp AS SELECT src, dst, idx FROM edge group by src, dst")
        cursor.execute("DROP TABLE edge;")
        cursor.execute("ALTER TABLE temp RENAME TO edge;")
    update_pbar(pbar)

    # 3. Merge.
    cursor.execute('CREATE INDEX edge_dst ON edge(dst);')
    cursor.execute('CREATE INDEX edge_src ON edge(src);')
    if mode == 0:
        cursor.execute('CREATE TABLE merged AS SELECT a.src as src, a.dst as mid, b.src as dst, a.idx as idx1, b.idx as idx2 FROM edge as a, edge as b WHERE a.dst = b.dst;')
    elif mode == 1:
        cursor.execute('CREATE TABLE merged AS SELECT a.src as src, a.dst as mid, b.dst as dst, a.idx as idx1, b.idx as idx2 FROM edge as a, edge as b WHERE a.dst = b.src;')
    update_pbar(pbar)

    # 4. De-duplicate 2-order edges.
    if sord:
        cursor.execute("CREATE TABLE temp AS SELECT src, mid, dst, idx1, idx2 FROM merged group by src, dst;")
        cursor.execute("DROP TABLE merged;")
        cursor.execute("ALTER TABLE temp RENAME TO merged;")
    update_pbar(pbar)

    # 5. To numpy.
    arrays = np.array(cursor.execute('SELECT src, mid, dst, idx1, idx2 FROM merged;').fetchall()).astype(np.int64)
    src, mid, dst, idx1, idx2 = arrays[:, 0], arrays[:, 1], arrays[:, 2], arrays[:, 3], arrays[:, 4]
    conn.close()
    os.remove(db_path)
    update_pbar(pbar, True)

    return src, mid, dst, idx1, idx2

def expansion_edge(edge_mini: pd.DataFrame):
    edge_mni_r = edge_mini.copy().rename(columns={'src': 'dst', 'dst': 'src'})
    edge_mni_u = pd.concat([edge_mini, edge_mni_r], axis=0)
    return edge_mni_u

def get_od2_graph(node_df: pd.DataFrame, edge_df: pd.DataFrame, efeat: Tensor, daylimit: float, mode: int, notqdm: bool):
    """Get second order graph.
    
    Args:
        node_df: The dataframe of nodes, with columns ['id', ...].
        edge_df: First order edges, with columns ['src', 'dst', 'time', ...].
        efeat: The edge features of first order edges, with shape (num_edges, d).
        daylimit: The time limit of second order edges.
    Returns:
        edge2_df: Second order edges, with columns ['src', 'mid', 'dst', 'time'].
        efeat2: The edge features of second order edges, with shape (num_edges, d).
    """

    # Get od2 data for rnn.
    edge_df_mini = edge_df[['src', 'dst']]
    src, mid, dst, idx1, idx2 = get_edge_od2(node_df, edge_df_mini, mode=mode, ford=True, sord=False, notqdm=notqdm)
    
    etime = edge_df['time'].to_numpy().astype(np.float32)
    etime = np.concatenate([etime, etime])
    valid = np.abs(etime[idx1] - etime[idx2]) < daylimit * 24 * 60 * 60
    valid = valid & (src != dst)
    src, dst, mid, idx1, idx2 = map(lambda x: x[valid], [src, dst, mid, idx1, idx2])

    etime= etime[idx2]
    efeat = torch.cat([efeat[idx1],  efeat[idx2]], dim=1).float()
    edge2_df = pd.DataFrame({'src': src, 'mid': mid, 'dst': dst, 'time': etime})

    return edge2_df, efeat

def get_struct(node_df: pd.DataFrame, edge_df: pd.DataFrame, dataset: str, daylimit: float, notqdm: bool):
    if notqdm:
        pbar = None
    else:
        pbar = tqdm(total=7, desc=f'[get_struct]{dataset.capitalize()}({daylimit:.1f}days)', leave=False)

    # Prepare data.
    nid, label = node_df['id'].to_numpy(), node_df['label'].to_numpy()
    edge_df_new = edge_df.copy()
    cnt = edge_df_new.groupby(['src', 'dst']).size().reset_index(name='cnt')['cnt'].to_numpy()
    edge_df_new = edge_df_new.sort_values(by=['time'])
    edge_dedup = edge_df_new.drop_duplicates(['src', 'dst'])
    src, dst, etime = edge_dedup['src'].to_numpy(), edge_dedup['dst'].to_numpy(), edge_dedup['time'].to_numpy()
    max_id = max(int(np.max(src)), int(np.max(dst)), int(np.max(nid)))
    update_pbar(pbar)

    # Get 1-order adj.
    create_csr = lambda data: sparse.coo_matrix((data, (src, dst)), shape=(max_id + 1, max_id + 1)).tocsr()
    adj = create_csr(np.ones_like(src))  # Adjacency matrix, eij is 1 if there is an edge from i to j.
    adj_path = create_csr(cnt)  # Adjacency matrix, eij is the number of edges from i to j.
    adj_time = create_csr(etime)  # Adjacency matrix, eij is the time of the first edge from i to j.
    update_pbar(pbar)

    # Get 2-order adj.
    adj2_path: sparse.csr_matrix = adj.dot(adj.T)
    adj2_time1: sparse.csr_matrix = adj_time.dot(adj.T)
    adj2_time2: sparse.csr_matrix = adj.dot(adj_time.T)
    def slice(x) -> sparse.csr_matrix:
        return x[nid, :].tocsc()[:, nid].tocsr()
    adj2_path, adj2_time1, adj2_time2 = map(slice, [adj2_path, adj2_time1, adj2_time2])
    update_pbar(pbar)

    adj2_path.data = adj2_path.data.astype(float)  # type: ignore
    def div_path(x) -> sparse.csr_matrix:
        if x.nnz == 0:
            return x
        assert adj2_path.nnz == x.nnz, f'{adj2_path.nnz} != {x.nnz}'
        x_copy = x.copy()
        x_copy.data = x.data / adj2_path.data
        return x_copy
    adj2_time1, adj2_time2 = div_path(adj2_time1), div_path(adj2_time2)
    update_pbar(pbar)

    adj2_delta: sparse.csr_matrix = abs(adj2_time1 - adj2_time2)
    update_pbar(pbar)

    # Encode struct.
    node_df['od1'] = adj_path.sum(axis=1)[nid]
    node_df['od1_dedup'] = adj.sum(axis=1)[nid]
    node_df['od2'] = adj2_path.sum(axis=1)
    def cnt_nonzero(x) -> np.matrix:
        y = x.astype(bool).astype(int).sum(axis=1)
        return y + 1e-6
    def get_mean(x: sparse.csr_matrix) -> np.ndarray:
        a = x.sum(axis=1).A.reshape(-1)
        b = cnt_nonzero(x).A.reshape(-1)
        return a / (b + 1e-6)
    node_df['od2_intime'] = get_mean(adj2_time1)
    node_df['od2_outtime'] = get_mean(adj2_time2)
    node_df['od2_delta'] = get_mean(adj2_delta)
    update_pbar(pbar)

    # Get pair_df.
    if adj2_delta.nnz > 0:
        adj2_valid = adj2_delta <= daylimit * 24 * 60 * 60
        adj2_valid = adj2_valid * (adj2_path > 0)
    else:
        adj2_valid = adj2_path
    adj2_valid_coo = sparse.coo_matrix(adj2_valid)
    src, dst = adj2_valid_coo.row, adj2_valid_coo.col
    pair_df = pd.DataFrame({'src': nid[src], 'dst': nid[dst], 'src_label': label[src], 'dst_label': label[dst]})
    update_pbar(pbar, True)

    return node_df, pair_df

def sample_edges(edge_df: pd.DataFrame, efeat: Tensor, num_sample: int):
    """Sample edges from edge_df and efeat."""

    # Sort by time.
    num_edges = len(edge_df)
    edge_df['idx'] = np.arange(num_edges)
    edge_df = edge_df.sort_values(by=['src', 'time'])
    efeat = efeat[edge_df['idx'].to_numpy()]

    # Sample edges.
    srcs = edge_df['src'].to_numpy()
    idx = torch.arange(num_edges)
    _, cnt = np.unique(srcs, return_counts=True)
    idx_list = torch.split(torch.arange(num_edges).long(), cnt.tolist())

    idx_list = [idx[:num_sample] for idx in idx_list]
    idx = torch.cat(idx_list, dim=0)
    edge_df = edge_df.iloc[idx]  # type: ignore
    efeat = efeat[idx]
    
    return edge_df, efeat

def assembling_feat(node_df: pd.DataFrame, edge_df: pd.DataFrame, nfeat: Tensor, efeat: Tensor):
    nshape, eshape = tuple(nfeat.shape), tuple(efeat.shape)
    keys = [k for k in node_df.columns if k not in ['id', 'label']]
    nfeat = torch.cat([nfeat, torch.from_numpy(node_df[keys].to_numpy()).float()], dim=1)
    func = lambda x: torch.from_numpy(x.to_numpy()).long()
    src, dst, nid = func(edge_df['src']), func(edge_df['dst']), func(node_df['id'])
    idx1, idx2 = find_idx(src, nid), find_idx(dst, nid)
    nfeat = torch.cat([nfeat, torch.randn(nfeat.shape[0], 1).float()], dim=1)
    efeat = torch.cat([efeat, nfeat[idx1], nfeat[idx2]], dim=1).float()
    # logging.debug(f'Assembling feat: nfeat={nshape}->{tuple(nfeat.shape)} efeat={eshape}->{tuple(efeat.shape)}')
    return nfeat, efeat

def sort_edge(edge_df: pd.DataFrame, efeat: Tensor):
    """Sort edges by time."""
    edge_df = edge_df.reset_index(drop=True)
    sort_idx = edge_df.sort_values(by=['src', 'time']).index
    sort_idx = sort_idx.to_numpy().astype(np.int64)
    edge_df = edge_df.iloc[sort_idx]
    sort_idx = torch.from_numpy(sort_idx).long()
    efeat = efeat[sort_idx]
    edge_df = edge_df.reset_index(drop=True)

    return edge_df, efeat

def dataset_info(node_df: pd.DataFrame, edge_df: pd.DataFrame, nfeat: Tensor, efeat: Tensor):
    num_pos = node_df[node_df['label'] == 1].shape[0]
    num_neg = node_df[node_df['label'] == 0].shape[0]
    pos_ratio = num_pos / (num_pos + num_neg)
    num_edges = edge_df.shape[0]
    num_nodes = node_df.shape[0]
    num_nfeat = tuple(nfeat.shape)
    num_efeat = tuple(efeat.shape)
    return f'nodes={num_nodes} pos={num_pos}({pos_ratio:.2%}) nfeat={num_nfeat} efeat={num_efeat}'

def sort_node(node_df, nfeat, edge_df):
    """Sort nodes by first edge time."""
    node_time_df = edge_df.sort_values(by=['time']).drop_duplicates(['src'], keep='first')[['src', 'time']]
    node_time_df = node_time_df.rename(columns={'src': 'id', 'time': 'time1'})
    node_df = node_df.merge(node_time_df, on='id', how='left')
    args_sort = np.argsort(node_df['time1'].to_numpy())
    node_df = node_df.iloc[args_sort]
    node_df = node_df.reset_index(drop=True)
    nfeat = nfeat[args_sort]
    return node_df, nfeat

@cache
def process_dataset(dataset: str, node_df: OPT_DF, edge_df: OPT_DF,
    sample1: int, sample2: int, time_dim: int, is_toundirect: bool,
    daylimit: float, mode: int, is_save_node: bool=True, notqdm: bool=False):
    """Gets the dataset for Cobifraud."""
    # Init pbar.
    if notqdm:
        pbar = None
    else:
        pbar = tqdm(total=11, desc=f'[df_to_dataset] Process dataset')

    # 1. Get efeat and edge_df, edge_df with columns ['src', 'dst', 'time].
    assert edge_df is not None
    node_df, edge_df, nfeat1, efeat1 = extract(node_df, edge_df)
    logging.debug(f'{str(dataset).capitalize()}: {dataset_info(node_df, edge_df, nfeat1, efeat1)}')
    update_pbar(pbar)
    
    # 2. Sort Node by time.
    node_df, nfeat1 = sort_node(node_df, nfeat1, edge_df)
    update_pbar(pbar)

    # 3. Get struct.
    node_df, pair_df = get_struct(node_df, edge_df, dataset, daylimit, notqdm)
    update_pbar(pbar)

    # 4. Save encoded node_df.
    if is_save_node:
        save_path = os.path.join(real_dir, 'readable')
        os.makedirs(save_path, exist_ok=True)
        node_df.to_csv(os.path.join(save_path, f'{dataset}_node.csv'), index=False)
    update_pbar(pbar)

    # 5. Assembling feat.
    nfeat1, efeat1 = assembling_feat(node_df, edge_df, nfeat1, efeat1)
    update_pbar(pbar)

    # 6. Sample 1-order edges.
    edge_df, efeat1 = sample_edges(edge_df, efeat1, sample1)
    update_pbar(pbar)

    # 7. Get 2-order datas.
    edge2_df, efeat2 = get_od2_graph(node_df, edge_df, efeat1, daylimit, mode=mode, notqdm=notqdm)
    update_pbar(pbar)

    # 8. Sample 2-order edges.
    edge2_df, efeat2 = sample_edges(edge2_df, efeat2, sample2)
    update_pbar(pbar)

    # 9. Direct to undirect.
    if is_toundirect:
        edge_df, efeat1 = to_undirect(edge_df, efeat1)
        edge_df, efeat1 = sample_edges(edge_df, efeat1, sample1)
    update_pbar(pbar)
    
    # 10. Encode time.
    edge_df, efeat1 = sort_edge(edge_df, efeat1)
    edge2_df, efeat2 = sort_edge(edge2_df, efeat2)
    get_etime = lambda edge_df : torch.from_numpy(edge_df['time'].to_numpy()).float()
    etime1, etime2 = get_etime(edge_df), get_etime(edge2_df)
    get_sections = lambda edge_df : np.unique(edge_df['src'].to_numpy(), return_counts=True)[1].tolist()
    sections, sections2 = get_sections(edge_df), get_sections(edge2_df)
    etime1 = encode_time(etime1, sections, time_dim)
    etime2 = encode_time(etime2, sections2, time_dim)
    update_pbar(pbar)

    # 11. Norm nfeat and efeat.
    func = lambda x: norm(x).nan_to_num(0)
    nfeat1, efeat1, efeat2, etime1, etime2 = func(nfeat1), func(efeat1), func(efeat2), func(etime1), func(etime2)
    update_pbar(pbar, True)
    
    return node_df, nfeat1, edge_df, efeat1, etime1, edge2_df, efeat2, etime2, pair_df

def get_dataset(args: Args, notqdm=False):
    """Get a Dataset by name."""
    # Read data from csv file.
    data_dir = join(real_dir, 'data_csv_norm')
    node, edge = read_node_edge(data_dir, args.dataset)

    # Process data.
    start_time = time.time()
    processed = process_dataset(args.dataset, node, edge, args.sample1, args.sample2, args.time_dim, args.direct_to_undirect, args.daylimit, args.mode, notqdm=notqdm)
    logging.debug(f'Process "{args.dataset}" finished, {time.time() - start_time:.2f}s, {mem_usage()}.')

    # Create dataset object.
    start_time = time.time()
    dataset = Dataset(*processed, args, notqdm=notqdm)
    logging.debug(f'Create "{args.dataset}" dataset finished, {time.time() - start_time:.2f}s, {mem_usage()}.')

    return dataset
    
if __name__ == '__main__':
    # Init logging.
    filename = os.path.join(real_dir, 'log', 'read_data.log')
    format = '%(asctime)s - %(process)d - %(levelname)s - %(funcName)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=filename, encoding='utf-8', format=format, datefmt=datefmt, level=logging.DEBUG)

    # names
    _names = ['yelpchi', 'wikipedia', 'elliptic', 'reddit', 'sichuan', 'mooc', 'amazon']
    # _names = ['sichuan']

    # Read data.
    for _name in _names:
        print(f'\nRead Dataset {str(_name).capitalize()}...')
        _args = Args(dataset=_name)
        _args.sample1 = 10
        _args.sample2 = 10
        dataset = get_dataset(_args)
        gc.collect()
