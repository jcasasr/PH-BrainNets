import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import logging

from Subject import Subject
import utils_gnn
from config import *


def array_to_graph(subject:Subject, data_type_structure:str=None, data_type_embbedings=[], num_classes:int=2, thr:float=0.0, logger=None):
    assert data_type_structure in MATRIX_TYPES, "Invalid type: {}".format(data_type_structure)

    # Get matrix (structure or topology)
    data = subject.get_matrix(data_type_structure)
    
    # Get target attribute (class)
    if num_classes==2:
        y = int(subject.get_mstype(type="binary"))
    else:
        y = int(subject.get_mstype(type="multiclass"))
    num_nodes = data.shape[0]

    # Adjancency matrix (A)
    num_edges = 0
    edge_index = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if data[i, j] >= thr:
                    edge_index.append([i, j])
                    edge_weight.append(data[i, j])
                    num_edges += 1

    if logger is not None:
        logger.debug("Using THR = {} -> number of edges = {} ({}%)".format(thr, num_edges, num_edges / (num_nodes * num_nodes) * 100))

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Node embeddings (X)
    node_embeddings = []
    for data_type_embbeding in data_type_embbedings:
        for metric in METRIC_LIST:
            key = data_type_embbeding +"-"+ metric
            node_embeddings.append(subject.get_metric(key))
    
    node_embeddings = np.array(node_embeddings)

    # duplicate embeddings (because we have duplicated the number of nodes in the super-adjacency matrix)
    if data_type_structure == "ML":
        node_embeddings = np.transpose(np.concatenate((node_embeddings, node_embeddings), axis=1))
    else:
        node_embeddings = np.transpose(node_embeddings)
    
    x = torch.tensor(node_embeddings, dtype=torch.float)
    
    # class label (y)
    y = torch.tensor([y], dtype=torch.long)
    
    # All together
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight, y=y)
    
    return data
