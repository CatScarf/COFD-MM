from typing import List, Tuple, Optional

class Args:
    def __init__(self, **kwargs):
        def set_attr(name, default):
            return default if name not in kwargs else kwargs[name]
            
        # Train args.
        self.device: str = set_attr('device', 'cuda:0')
        self.dataset: str = set_attr('dataset', 'sichuan')
        self.seed: int = set_attr('seed', 2023)
        self.epochs: int = set_attr('epochs', 100)
        self.lr: float = set_attr('lr', 0.005)
        
        self.tolerance: int = set_attr('tolerance', 3)
        self.midinit: int = set_attr('midinit', 20)

        # Data args.
        self.batch_size: int = set_attr('batch_size', 200000)
        self.balance_split: bool = set_attr('balance_split', False)
        self.split_ratio: List[float] =  set_attr('split_ratio', [0.8, 0.1, 0.1])
        self.time_dim: int = set_attr('time_dim', 8)
        self.direct_to_undirect: bool = set_attr('direct_to_undirect', False)

        self.sample1: int = set_attr('sample1', 50)  # First-order neighbor maximum time step.
        self.sample2: int = set_attr('sample2', 20)  # Second-order neighbor maximum time step.
        self.sample3: int = set_attr('sample3', 20)
        self.sample4: int = set_attr('sample4', 20)
        self.daylimit: float = set_attr('daylimit', 7.0)

        self.mode: int = set_attr('mode', 0)  # 0, 1

        # Loss args.
        self.lossfn1: str = set_attr('lossfn', 'mse')  # mse, ce, focal
        self.lossfn3: str = set_attr('lossfn', 'focal')  # mse, ce, focal
        self.loss_weight: float = set_attr('loss_weight', 0.9)  # for ce loss
        self.alpha: float = set_attr('alpha', 0.5)  # for focal loss
        self.gamma: float = set_attr('gamma', 0.25)  # for focal loss

        # Model args
        self.emb_dim: int = set_attr('emb_dim', 4)
        self.encoder: str = set_attr('encoder', 'lstm')  # lstm only
        self.lstm_layers: int = set_attr('lstm_layers', 3)
        self.mlp_layers: List[int] = set_attr('mlp_layers', [1024, 1024, 512, 256, 128])
        self.norm: bool = set_attr('norm', True)
        
        self.update_feat: bool = set_attr('update_feat', True)
        self.embmode: str = set_attr('embmode', 'zeros')  # randn or zeros


    def __str__(self):
        kwargs: List[Tuple[str, str]] = list((str(k), str(v)) for k, v in dict(self.__dict__).items())
        return ', '.join(v.replace(', ', '-') for k, v in kwargs)

    def dict_str(self):
        return dict(self.__dict__)

    def keys(self):
        return list(self.__dict__.keys())

def get_args(dataset: Optional[str] = None):
    best = None
    if dataset == 'sichuan':
        best = {'device': 'cuda:0', 'dataset': 'sichuan', 'seed': 27, 'epochs': 50, 'lr': 0.01, 'tolerance': 10, 'midinit': 2, 'batch_size': 100000, 'balance_split': False, 'split_ratio': (0.6, 0.2, 0.2), 'time_dim': 4, 'direct_to_undirect': False, 'sample1': 90, 'sample2': 20, 'sample3': 10, 'sample4': 10, 'daylimit': 2, 'mode': 0, 'lossfn1': 'mse', 'lossfn3': 'focal', 'loss_weight': 0.7728173313240538, 'alpha': 0.8336981924640765, 'gamma': 0.45401494155151967, 'emb_dim': 32, 'encoder': 'lstm', 'lstm_layers': 1, 'mlp_layers': (1024, 512, 256, 128, 64), 'norm': True, 'update_feat': True, 'embmode': 'randn'}
    elif dataset == 'sichuan_fast':
        best = {'device': 'cuda:2', 'dataset': 'sichuan', 'seed': 27, 'epochs': 100, 'lr': 0.001, 'tolerance': 10, 'midinit': -1, 'batch_size': 200000, 'balance_split': False, 'split_ratio': (0.6, 0.2, 0.2), 'time_dim': 8, 'direct_to_undirect': False, 'sample1': 10, 'sample2': 10, 'sample3': 10, 'sample4': 10, 'daylimit': 3, 'lossfn1': 'mse', 'lossfn3': 'focal', 'loss_weight': 0.5873709407657591, 'alpha': 0.7318461966740528, 'gamma': 0.99596927546076, 'emb_dim': 32, 'encoder': 'lstm', 'lstm_layers': 4, 'mlp_layers': (2048, 1024, 512, 256, 128, 64), 'norm': True, 'update_feat': True, 'embmode': 'randn'}
    elif dataset == 'amazon':
        best = {'device': 'cuda:1', 'dataset': 'amazon', 'seed': 55, 'epochs': 100, 'lr': 0.005, 'tolerance': 14, 'midinit': 9, 'batch_size': 200000, 'balance_split': False, 'split_ratio': (0.6, 0.2, 0.2), 'time_dim': 2, 'direct_to_undirect': False, 'sample1': 20, 'sample2': 10, 'sample3': 10, 'sample4': 10, 'daylimit': 1, 'lossfn1': 'mse', 'lossfn3': 'focal', 'loss_weight': 0.6682525318755957, 'alpha': 0.8314901563963462, 'gamma': 0.337641921306962, 'emb_dim': 4, 'encoder': 'lstm', 'lstm_layers': 3, 'mlp_layers': (1024, 512, 256, 128, 64), 'norm': True, 'update_feat': True, 'embmode': 'zeros'}
    elif dataset == 'yelpchi':
        best = {'device': 'cuda:0', 'dataset': 'yelpchi', 'seed': 70, 'epochs': 100, 'lr': 0.01, 'tolerance': 8, 'midinit': 18, 'batch_size': 200000, 'balance_split': False, 'split_ratio': (0.6, 0.2, 0.2), 'time_dim': 2, 'direct_to_undirect': True, 'sample1': 70, 'sample2': 40, 'sample3': 40, 'sample4': 10, 'daylimit': 1, 'lossfn1': 'mse', 'lossfn3': 'focal', 'loss_weight': 0.5351388747121317, 'alpha': 0.711429279033398, 'gamma': 0.9020717973496053, 'emb_dim': 128, 'encoder': 'lstm', 'lstm_layers': 2, 'mlp_layers': (1024, 512, 256, 128, 64), 'norm': True, 'update_feat': True, 'embmode': 'randn'}

    if best is not None:
        return Args(**best)
    else:
        return Args(dataset=dataset)