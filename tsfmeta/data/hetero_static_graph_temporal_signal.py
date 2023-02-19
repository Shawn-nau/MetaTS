import torch
import numpy as np
from typing import List, Union,Tuple
import pandas as pd

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch
from collections.abc import Mapping, Sequence
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader,Dataset

Edge_Index = List[Union[np.ndarray, None]]
Edge_Weight = List[Union[np.ndarray, None]]
NNets = Union[int, None]
Node_Features_Timevarying = List[Union[np.ndarray, None]]
Node_Features_Static = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Target_features = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


                        
class HeteroStaticGraphTemporalSignal(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,        
        unknown_features: Node_Features_Timevarying,
        static_features: Node_Features_Static,
        targets: Targets,
        targets_mask: Targets,
        target_features:Target_features,
        scales:List[Union[np.ndarray, None]],
        tdx:List[Union[np.ndarray, None]],
        idx:List[Union[np.ndarray, None]],
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight        
        self.unknown_features = unknown_features
        self.static_features = static_features
        self.targets = targets
        self.targets_mask = targets_mask
        self.target_features = target_features
        self.additional_feature_keys = []
        self.scales = scales
        self.tdx = tdx
        self.idx = idx
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.unknown_features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.unknown_features)

    def _get_edge_index(self, time_index: int):
        if self.edge_index[time_index] is None:
            return self.edge_index[time_index]
        else:
            return torch.LongTensor(self.edge_index[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weight[time_index] is None:
            return self.edge_weight[time_index]
        else:
            return torch.FloatTensor(self.edge_weight[time_index])

    def _get_features(self, time_index: int):
        if self.unknown_features[time_index] is None:
            return self.unknown_features[time_index]
        else:
            return torch.FloatTensor(
                    np.nan_to_num( self.unknown_features[time_index])
                    )

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(
                        np.nan_to_num( self.targets[time_index])
                        )
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(
                        np.nan_to_num( self.targets[time_index])
                        )

    def _get_target_mask(self, time_index: int):
        if self.targets_mask[time_index] is None:
            return self.targets_mask[time_index]
        else:
            if self.targets_mask[time_index].dtype.kind == "i":
                return torch.LongTensor(
                        np.nan_to_num( self.targets_mask[time_index])
                        )
            elif self.targets_mask[time_index].dtype.kind == "f":
                return torch.FloatTensor(
                        np.nan_to_num( self.targets_mask[time_index])
                        )

    def _get_target_features(self, time_index: int):
        if self.target_features[time_index] is None:
            return self.target_features[time_index]
        else:
            return torch.FloatTensor(
                    np.nan_to_num( self.target_features[time_index])
                    )

    def _get_static_features(self, time_index: int):
        if self.static_features[time_index] is None:
            return self.target_features[time_index]
        else:
            return torch.FloatTensor(
                    np.nan_to_num( self.static_features[time_index],nan = 0)
                    )

    def _get_target_scale(self, time_index: int):
        if self.scales[time_index] is None:
            return self.scales[time_index]
        else:
            return torch.FloatTensor(
                    np.nan_to_num( self.scales[time_index])
                    )

    def _get_target_tdx(self, time_index: int):
        if self.tdx[time_index] is None:
            return self.tdx[time_index]
        else:
            return torch.FloatTensor(self.tdx[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __get_item__(self, time_index: int):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index(time_index)
        edge_weight = self._get_edge_weight(time_index)
        y = self._get_target(time_index)
        mask = self._get_target_mask(time_index)
        z = self._get_target_features(time_index)
        scale = self._get_target_scale(time_index)
        static_f = self._get_static_features(time_index)
        
        additional_features = self._get_additional_features(time_index)
        snapshot = HeteroData()
        snapshot['node'].x = x
        snapshot['node'].y = y
        snapshot['node'].mask = mask
        snapshot['node'].z = z
        snapshot['node'].scale = scale
        snapshot['tdx'] = self.tdx[time_index]
        snapshot['idx'] = self.idx[time_index]        
        
        snapshot['nodenet'].edge_index = edge_index        
        snapshot['nodenet'].edge_attr = edge_weight
        
        
        snapshot['node'].f = static_f
 
        return snapshot

    def __next__(self):
        if self.t < len(self.unknown_features):
            snapshot = self.__get_item__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
    
    def __len__(self):
        """Compute length."""
        return len(self.unknown_features)
    
    @staticmethod   
    def _collate_fn(
        batches: List[HeteroData]
        ) -> HeteroData:
        
        return Batch.from_data_list(batches)

    def to_dataloader(
        self, train: bool = True, batch_size: int = 1, batch_sampler: Union[Sampler, str] = None, **kwargs
    ) -> DataLoader:
        
        """
        Get dataloader from dataset.
        """
        data = iter(self)
        
        dataset = [next(data) for j in range(len(self))]
        
        default_kwargs = dict(
            shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        return DataLoader(
            dataset,
            **kwargs,
        ) 

 


def temporal_signal_split(
    data_iterator, train_ratio: float = 0.8, H : float = 1
) -> Tuple[HeteroStaticGraphTemporalSignal, HeteroStaticGraphTemporalSignal]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """

    train_snapshots = int(train_ratio * data_iterator.snapshot_count)

    train_iterator = HeteroStaticGraphTemporalSignal(
        data_iterator.edge_index[0:(train_snapshots-H+1)],
        data_iterator.edge_weight[0:(train_snapshots-H+1)],        
        data_iterator.unknown_features[0:(train_snapshots-H+1)],
        data_iterator.static_features[0:(train_snapshots-H+1)],
        
        data_iterator.targets[0:(train_snapshots-H+1)],
        data_iterator.targets_mask[0:(train_snapshots-H+1)],
        data_iterator.target_features[0:(train_snapshots-H+1)],
        
        data_iterator.scales[0:(train_snapshots-H+1)],
        data_iterator.tdx[0:(train_snapshots-H+1)],
        data_iterator.idx[0:(train_snapshots-H+1)],
        
    )

    test_iterator = HeteroStaticGraphTemporalSignal(
        data_iterator.edge_index[train_snapshots:],
        data_iterator.edge_weight[train_snapshots:],
        
        data_iterator.unknown_features[train_snapshots:],
        data_iterator.static_features[train_snapshots:],
        
        data_iterator.targets[train_snapshots:],
        data_iterator.targets_mask[train_snapshots:],
        data_iterator.target_features[train_snapshots:],
        
        data_iterator.scales[train_snapshots:],
        data_iterator.tdx[train_snapshots:],
        data_iterator.idx[train_snapshots:],
        
    )

    return train_iterator, test_iterator



def HSGT_concat(
    data_iterator: List[HeteroStaticGraphTemporalSignal]
) -> HeteroStaticGraphTemporalSignal:
    
    edge_index = []
    edge_weight = []
    
    unknown_features = []
    static_features = []
    targets= []
    targets_mask = []
    target_features = []
    scales =[]
    tdx = []
    idx = []
    
    N =  len(data_iterator)
    
    for i in range(N):
        edge_index += data_iterator[i].edge_index
        edge_weight += data_iterator[i].edge_weight
        unknown_features += data_iterator[i].unknown_features
        static_features += data_iterator[i].static_features
        targets += data_iterator[i].targets
        targets_mask += data_iterator[i].targets_mask
        target_features += data_iterator[i].target_features
        scales += data_iterator[i].scales
        tdx += data_iterator[i].tdx
        idx += data_iterator[i].idx
    
    concated_data = HeteroStaticGraphTemporalSignal(
        edge_index = edge_index,
        edge_weight = edge_weight,
        
        unknown_features = unknown_features,
        static_features =static_features,
        
        targets = targets,
        targets_mask = targets_mask,
        target_features = target_features,
        
        scales = scales,
        tdx = tdx,
        idx = idx,
        
    )
   
    return concated_data


