import os
import time
from typing import Optional


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

def load_dataset(name: str):
    path = os.path.join(os.path.dirname(__file__), 'data_csv_norm')
    node_path = os.path.join(path, name + '_node.csv')
    edge_path = os.path.join(path, name + '_edge.csv')
    node_df = pd.read_csv(node_path) if os.path.exists(node_path) else None
    edge_df = pd.read_csv(edge_path)
    return node_df, edge_df

def process(node_df: Optional[pd.DataFrame], edge_df: pd.DataFrame):
    if node_df is not None:
        x = node_df.drop(['id', 'label'], axis=1).to_numpy()
        y = node_df['label'].to_numpy()
    else:
        x = edge_df.drop(['src', 'dst', 'label'], axis=1).to_numpy()
        y = edge_df['label'].to_numpy()
    return x, y

def split (x: np.ndarray, y: np.ndarray, ratio=0.8, seed = 0):
    np.random.seed(seed)
    n = len(x)
    idx = np.random.permutation(n)
    train_idx = idx[:int(n * ratio)]
    test_idx = idx[int(n * ratio):]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return x_train, y_train, x_test, y_test

def train_xgb(x_train, y_train, x_test, y_test, params):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    bst = xgb.train(params, dtrain, num_boost_round=100)
    y_pred = bst.predict(dtest)
    y_pred = (y_pred > 0.5).astype(int)
    return y_pred

if __name__ == '__main__':
    for name in ['sichuan', 'mooc', 'amazon', 'yelpchi']:
        node_df, edge_df = load_dataset(name)
        x, y = process(node_df, edge_df)
        start_time = time.time()
        x_train, y_train, x_test, y_test = split(x, y)
        y_pred = train_xgb(x_train, y_train, x_test, y_test, {})
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred)
        print(name, x_train.shape, f'pos: {sum(y)/len(y):.2%}', f'auc:{auc:.2%} f1:{f1:.2%} duration: {time.time()-start_time:.2f}s')