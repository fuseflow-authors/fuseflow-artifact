import inspect
import torch
import math
from numpy.core.shape_base import block
import torch_geometric
from torch_geometric.datasets import Planetoid, TUDataset, QM9, CitationFull, Amazon, AttributedGraphDataset, KarateClub, MovieLens100K
from torch_geometric.utils import to_scipy_sparse_matrix
from collections import defaultdict
import scipy.sparse as sp
import random
import numpy as np
import os 
from sam.onyx.generate_matrices import *
from sam.sim.test.test import *
# Load a dataset (e.g., Cora from Planetoid dataset)
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset  
from ogb.graphproppred import PygGraphPropPredDataset
#from ogb.graphproppred import PygGraph


np.random.seed(4)
random.seed(4)

def check_gold(gold_file, result_file):
    gold_vals = []
    result_vals = []

    with open(gold_file, "r") as f:
        for line in f:
            gold_vals.append(float(line))

    with open(result_file, "r") as f:
        for line in f:
            result_vals.append(float(line))
    
    # print(result_vals, gold_vals)
    assert(len(gold_vals) == len(result_vals))
    # gold_vals.sort()
    # result_vals.sort()
    diff = []
    for i in range(len(gold_vals)):
        # if gold_vals[i] != result_vals[i]:
        if not math.isclose(gold_vals[i], result_vals[i], rel_tol=1e-5, abs_tol=0.0):
            diff.append((i, gold_vals[i]-result_vals[i])) 
    
    print(diff)
    return False if len(diff) > 0 else True
        
    

# Mapping of dataset names to their corresponding PyTorch Geometric dataset classes
DATASETS = {
    'Planetoid': Planetoid,
    'TUDataset': TUDataset,
    'QM9': QM9,
    'CitationFull': CitationFull,
    'Amazon': Amazon,
    'KarateClub': KarateClub,
    'AttributedGraphDataset': AttributedGraphDataset,
    'MovieLens100K': MovieLens100K,
    'ogbn': NodePropPredDataset,      # For node classification (ogbn-arxiv, ogbn-mag, etc.)
    'ogbl': LinkPropPredDataset,      # For link prediction
    'ogbg': PygGraphPropPredDataset,     # For graph classification
}

def convert_adjacency_matrix_efficient(data):
    """
    More memory-efficient conversion that works directly with edge_index
    without creating a full scipy sparse matrix
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Work directly with edge_index (already in COO format)
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Create sparse matrix directly from COO data
    row_indices = edge_index[0]
    col_indices = edge_index[1]

    # Use ones as values (unweighted graph)
    values = np.ones(len(row_indices), dtype=np.float32)  # Use float32 instead of float64

    # Create sparse matrix in CSR format (more memory efficient than COO)
    sparse_matrix = sp.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    return sparse_matrix

def convert_adjacency_matrix_chunked(data, chunk_size=10000):
    """
    Process adjacency matrix in chunks to reduce memory usage
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    num_edges = edge_index.shape[1]

    # Process edges in chunks
    row_list = []
    col_list = []
    data_list = []

    for start_idx in range(0, num_edges, chunk_size):
        end_idx = min(start_idx + chunk_size, num_edges)

        chunk_rows = edge_index[0, start_idx:end_idx]
        chunk_cols = edge_index[1, start_idx:end_idx]
        chunk_data = np.ones(len(chunk_rows), dtype=np.float32)

        row_list.append(chunk_rows)
        col_list.append(chunk_cols)
        data_list.append(chunk_data)

    # Concatenate all chunks
    all_rows = np.concatenate(row_list)
    all_cols = np.concatenate(col_list)
    all_data = np.concatenate(data_list)

    # Create sparse matrix
    sparse_matrix = sp.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )

    return sparse_matrix

def create_fiber_tree_efficient(edge_index, num_nodes=None, mode_ordering=[0, 1], chunk_size=300000):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Apply mode ordering - if mode_ordering is [1, 0], transpose the matrix
    if mode_ordering == [1, 0]:
        # Swap row and column indices
        indices = edge_index[[1, 0]]
    else:
        indices = edge_index

    row_indices = indices[0]
    col_indices = indices[1]
    num_edges = len(row_indices)

    if num_nodes is None:
        num_nodes = max(np.max(row_indices), np.max(col_indices)) + 1

    fiber_tree = defaultdict(lambda: {"seg": [], "crd": []})
    fiber_tree_vals = []

    # Process in chunks to avoid memory issues
    unique_rows_set = set()

    # First pass: collect unique rows efficiently
    for start_idx in range(0, num_edges, chunk_size):
        end_idx = min(start_idx + chunk_size, num_edges)
        chunk_rows = row_indices[start_idx:end_idx]
        unique_rows_set.update(chunk_rows)

    unique_rows = sorted(list(unique_rows_set))

    # Initialize fiber tree structure
    fiber_tree["0"]["seg"] = [0, len(unique_rows)]
    fiber_tree["0"]["crd"] = unique_rows

    # Second pass: process columns for each unique row
    row_to_cols = defaultdict(list)

    # Group edges by row efficiently
    for start_idx in range(0, num_edges, chunk_size):
        end_idx = min(start_idx + chunk_size, num_edges)
        chunk_rows = row_indices[start_idx:end_idx]
        chunk_cols = col_indices[start_idx:end_idx]

        for r, c in zip(chunk_rows, chunk_cols):
            row_to_cols[r].append(c)

    # Build fiber tree level 1
    for r in unique_rows:
        cols = sorted(row_to_cols[r])
        fiber_tree["1"]["seg"].append(len(fiber_tree["1"]["crd"]))
        fiber_tree["1"]["crd"].extend(cols)
        fiber_tree_vals.extend([1.0] * len(cols))  # Assuming unweighted

    fiber_tree["1"]["seg"].append(len(fiber_tree["1"]["crd"]))

    # Calculate sparsity
    total_possible = num_nodes * num_nodes
    sparsity = len(fiber_tree_vals) / total_possible
    print(f"Sparsity: {sparsity:.6f}")

    return (num_nodes, num_nodes), fiber_tree, fiber_tree_vals

def backup_create_fiber_tree_efficient(edge_index, num_nodes=None, chunk_size=50000):
    """
    Create fiber tree directly from edge_index without intermediate sparse matrix
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    row_indices = edge_index[0]
    col_indices = edge_index[1]
    num_edges = len(row_indices)

    if num_nodes is None:
        num_nodes = max(np.max(row_indices), np.max(col_indices)) + 1

    fiber_tree = defaultdict(lambda: {"seg": [], "crd": []})
    fiber_tree_vals = []

    # Process in chunks to avoid memory issues
    processed_edges = 0
    unique_rows_set = set()

    # First pass: collect unique rows efficiently
    for start_idx in range(0, num_edges, chunk_size):
        end_idx = min(start_idx + chunk_size, num_edges)
        chunk_rows = row_indices[start_idx:end_idx]
        unique_rows_set.update(chunk_rows)

    unique_rows = sorted(list(unique_rows_set))

    # Initialize fiber tree structure
    fiber_tree["0"]["seg"] = [0, len(unique_rows)]
    fiber_tree["0"]["crd"] = unique_rows

    # Second pass: process columns for each unique row
    row_to_cols = defaultdict(list)

    # Group edges by row efficiently
    for start_idx in range(0, num_edges, chunk_size):
        end_idx = min(start_idx + chunk_size, num_edges)
        chunk_rows = row_indices[start_idx:end_idx]
        chunk_cols = col_indices[start_idx:end_idx]

        for r, c in zip(chunk_rows, chunk_cols):
            row_to_cols[r].append(c)

    # Build fiber tree level 1
    for r in unique_rows:
        cols = sorted(row_to_cols[r])
        fiber_tree["1"]["seg"].append(len(fiber_tree["1"]["crd"]))
        fiber_tree["1"]["crd"].extend(cols)
        fiber_tree_vals.extend([1.0] * len(cols))  # Assuming unweighted

    fiber_tree["1"]["seg"].append(len(fiber_tree["1"]["crd"]))

    # Calculate sparsity
    total_possible = num_nodes * num_nodes
    sparsity = len(fiber_tree_vals) / total_possible
    print(f"Sparsity: {sparsity:.6f}")

    return (num_nodes, num_nodes), fiber_tree, fiber_tree_vals

def load_dataset_memory_efficient(dataset_name, root_dir, name, use_mmap=True):
    """
    Memory-efficient dataset loading with optional memory mapping
    """
    if dataset_name in DATASETS:
        DatasetClass = DATASETS[dataset_name]

        if dataset_name == 'KarateClub':
            dataset = DatasetClass()
            return dataset[0], dataset.num_classes
        elif dataset_name.startswith('ogb'):
            # Handle OGB datasets with memory efficiency
            dataset = DatasetClass(name=name, root=root_dir)

            # Handle different OGB dataset types
            # Link prediction datasets (ogbl) return just graph dict, no labels
            # Node prediction datasets (ogbn) return (graph, labels) tuple
            if dataset_name == 'ogbl':
                # Link prediction: dataset[0] returns just the graph dict
                graph = dataset[0]
                labels = None  # Link prediction doesn't have node labels
            else:
                # For large datasets, consider using memory mapping
                if use_mmap and hasattr(dataset, 'data'):
                    # Access data without loading everything into memory at once
                    graph, labels = dataset[0]
                else:
                    graph, labels = dataset[0]

            # Convert OGB format to PyG format efficiently
            data = torch_geometric.data.data.Data()

            # Handle node features efficiently
            if 'node_feat' in graph:
                node_feat = graph['node_feat']
                if isinstance(node_feat, np.ndarray) and node_feat.dtype == np.float64:
                    # Convert to float32 to save memory
                    node_feat = node_feat.astype(np.float32)
                data.x = torch.from_numpy(node_feat)
            else:
                if name == 'ogbn-mag':
                    data.x = torch.from_numpy(graph['node_feat_dict']['paper'])
                else:
                    print("Found no X data")
                    data.x = None

            # Handle edge index efficiently
            # edge_index = graph['edge_index']
            data.edge_index = torch.from_numpy(graph['edge_index']) if 'edge_index' in graph else torch.from_numpy(graph['edge_index_dict'][('author', 'writes', 'paper')])
            # if isinstance(edge_index, np.ndarray) and edge_index.dtype != np.int64:
            #     edge_index = edge_index.astype(np.int64)
            # data.edge_index = torch.from_numpy(edge_index)

            # Handle edge features
            data.edge_attr = graph['edge_feat'] if 'edge_feat' in graph else None
            data.y = labels
            data.num_nodes = graph['num_nodes'] if 'num_nodes' in graph else None

            return data, dataset.num_classes
        else:
            # Handle standard PyG datasets
            dataset = DatasetClass(root=root_dir, name=name)
            return dataset[0], dataset.num_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def convert_node_features_efficient(node_features):
    """
    More memory-efficient node feature conversion
    """
    if isinstance(node_features, torch.Tensor):
        # Convert to numpy with appropriate dtype
        if node_features.dtype == torch.float64:
            node_features = node_features.float()  # Convert to float32
        node_features_numpy = node_features.cpu().numpy()
    else:
        node_features_numpy = node_features

    # If features are mostly sparse, convert to sparse format
    if node_features_numpy is not None:
        sparsity = np.count_nonzero(node_features_numpy) / node_features_numpy.size
        print(f"Node feature sparsity: {sparsity:.4f}")

        if sparsity < 0.1:  # If less than 10% non-zero, use sparse
            sparse_features = sp.csr_matrix(node_features_numpy, dtype=np.float32)
            print(f"Converted to sparse: {sparse_features.shape}, density: {sparse_features.nnz/sparse_features.size:.4f}")
            return sparse_features

    return node_features_numpy.astype(np.float32)

# Updated main conversion function
def process_graph_memory_efficient(data, use_chunked=True, chunk_size=50000):
    """
    Process graph data with memory efficiency in mind
    """
    print(f"Processing graph with {data.num_nodes} nodes and {data.edge_index.shape[1]} edges")

    # Option 1: Work directly with edge_index (most memory efficient)
    if use_chunked:
        shape, fiber_tree, fiber_tree_vals = create_fiber_tree_efficient(
            data.edge_index, data.num_nodes, chunk_size
        )
        return shape, fiber_tree, fiber_tree_vals
    else:
        # Option 2: Use more efficient sparse matrix conversion
        sparse_matrix = convert_adjacency_matrix_efficient(data)
        # return create_fiber_tree(sparse_matrix)
        return create_fiber_tree_efficient(sparse_matrix)

def load_dataset(dataset_name, root_dir, name):
    if dataset_name in DATASETS:
        DatasetClass = DATASETS[dataset_name]

        if dataset_name == 'KarateClub':
            dataset = DatasetClass()
            return dataset[0], dataset.num_classes
        elif dataset_name.startswith('ogb'):
            # Handle OGB datasets
            dataset = DatasetClass(name=name, root=root_dir)
            graph, labels = dataset[0]  # Get the graph and labels

            # Convert OGB format to PyG format
            data = torch_geometric.data.data.Data()
            data.x = torch.from_numpy(graph['node_feat']) if 'node_feat' in graph else None
            data.edge_index = torch.from_numpy(graph['edge_index']) if 'edge_index' in graph else torch.from_numpy(graph['edge_index_dict'][('author', 'writes', 'paper')])
            data.edge_attr = graph['edge_feat'] if 'edge_feat' in graph else None
            data.y = labels
            data.num_nodes = graph['num_nodes'] if 'num_nodes' in graph else None

            # Get number of classes
            if dataset_name == 'ogbn':
                num_classes = dataset.num_classes
            else:
                num_classes = len(torch.unique(labels)) if labels is not None else 1

            return data, dataset.num_classes
        else:
            # Handle standard PyG datasets
            dataset = DatasetClass(root=root_dir, name=name)
            return dataset[0], dataset.num_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Function to load dataset based on dataset name
def backup_load_dataset(dataset_name, root_dir, name):
    if dataset_name in DATASETS:
        DatasetClass = DATASETS[dataset_name]
        if dataset_name != 'KarateClub':
            dataset = DatasetClass(root=root_dir, name=name)
        elif dataset_name == 'KarateClub':
            dataset = DatasetClass()
        return dataset[0], dataset.num_classes  # Assuming we want the first graph in the dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Function to create fiber tree structure from sparse matrix
def convert_adjacency_matrix(data):
    # Convert edge_index to a sparse matrix
    sparse_matrix = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    return sparse_matrix

def convert_node_features(node_features):
    # Convert to NumPy array
    node_features_numpy = node_features.numpy()
    # print(node_features_numpy.shape)
    # Convert to sparse SciPy matrix (CSR format)
    # node_features_sparse = sp.coo_matrix(node_features_numpy)
    # print(node_features_sparse.shape)'
    # print("Node feature ", node_features_sparse.nnz/(node_features_sparse.shape[0]*node_features_sparse.shape[1]), node_features_sparse.shape)
    return node_features_numpy

def create_random_weights(rows, cols, density):
    # Generate random sparse matrix in COO format
    random_matrix = sp.random(rows, cols, density=density, format='coo', dtype=np.float64)
    print("rand ashape ", random_matrix.shape)
    # return create_fiber_tree(random_matrix)
    return create_fiber_tree_efficient(random_matrix)

def create_dense_bias(rows):
    bias_vector = np.random.rand(rows)
    return create_fiber_row(bias_vector)
    
def create_fiber_row(row):
    # Assuming the vector is dense
    # Otherwise need coord values and rows seperately
    fiber_tree = defaultdict(lambda: {"seg": [], "crd": []})
    fiber_tree["0"]["seg"] = [0, len(row)]
    fiber_tree["0"]["crd"] = list(range(len(row)))
    return fiber_tree, row

def create_sparse_row(row):
    fiber_tree = defaultdict(lambda: {"seg": [], "crd": []})
    fiber_tree["0"]["crd"] = []
    fiber_tree_vals = []
    for i, r in enumerate(row):
        if r != 0:
            fiber_tree["0"]["crd"].append(i)
            fiber_tree_vals.append(r)
    fiber_tree["0"]["seg"] = [0, len(fiber_tree_vals)]
    return [len(row)], fiber_tree, fiber_tree_vals


def create_fiber_tree(sparse_matrix):
    # Extract row and col indices, and values from sparse matrix
    row, col = sparse_matrix.nonzero()
    vals = sparse_matrix.data
    zipped_list = list(zip(row, col, vals))
    zipped_list.sort(key=lambda x: (x[0], x[1]))
    # Unzip the sorted list back into two lists
    row, col, vals = zip(*zipped_list)
    row, col, vals = ([int(x) for x in list(row)], col, vals)       
    fiber_tree = defaultdict(lambda: {"seg": [], "crd": []})
    fiber_tree_vals = []  # Separate list to store values
    unique_rows, segs = np.unique(row, return_index=True)
    fiber_tree["0"]["seg"] = [0, len(unique_rows.tolist())]
    # fiber_tree["1"]["seg"] = segs.tolist()
    fiber_tree["0"]["crd"] = unique_rows.tolist()
    # For each row, compress the column indices
    for i, r in enumerate(unique_rows):
        row_indices = np.where(row == r)[0]  # Get the row indices where row matches r
        # Ensure row_indices is an integer array
        row_indices = row_indices.astype(int)
        # Ensure col is a NumPy array
        col = np.array(col)
        vals = np.array(vals)
        # Use the row_indices to index col
        cols = col[row_indices]
        # row_indices = np.where(row == r)[0]
        # cols = col[row_indices]
        fiber_tree["1"]["seg"].append(len(fiber_tree['1']["crd"]))
        fiber_tree["1"]["crd"].extend(cols.tolist())
        fiber_tree_vals.extend(vals[row_indices].tolist())  # Store values separately    
    fiber_tree["1"]["seg"].append(len(fiber_tree['1']["crd"]))
    unique_rows = len(unique_rows) 
    assert len(fiber_tree["1"]["seg"]) == unique_rows+1
    assert unique_rows == fiber_tree["0"]["seg"][-1]
    assert unique_rows == len(fiber_tree["0"]["crd"])
    assert len(fiber_tree["1"]["crd"]) == len(fiber_tree_vals) and fiber_tree["1"]["seg"][-1] == len(fiber_tree_vals) 
    print(f"sparsities are {len(fiber_tree_vals)/(sparse_matrix.get_shape()[0]*sparse_matrix.get_shape()[1])}")
    return sparse_matrix.get_shape(), fiber_tree, fiber_tree_vals

# Main function to handle dataset loading and fiber tree creation
def data_gen(dataset_name, root_dir, name, xname=None, adjname=None, path=None, adjmode=[0, 1], xmode=[0, 1]):
    #in_channels = dataset.num_features
    #hidden_channels = 127
    #out_channels = dataset.num_classes
    data, _ = load_dataset_memory_efficient(dataset_name, root_dir, name)
    # data_features = convert_node_features(data.x)
    if type(data.x) == torch.Tensor:
        features = data.x.numpy()
    else:
        features = data.x

    # Convert edge_index to adjacency matrix for MatrixGenerator
    edge_index = data.edge_index.numpy()
    num_nodes = features.shape[0]
    # Create sparse adjacency matrix from edge_index
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0], edge_index[1]] = 1.0

    return MatrixGenerator(xname, features.shape, mode_ordering=xmode, sparsity=0.1, dump_dir=os.getcwd() + path, tensor=features, format="UNC"), MatrixGenerator(adjname, adj.shape, mode_ordering=adjmode, dump_dir=os.getcwd() + path, tensor=adj, format="CSF")


def weight_gen(feature_list, density, mode="random", num_linears=2):
    weight_bias_list = [] 
    if mode == "random":
        for i in range(len(feature_list[:-1])):
            for j in range(num_linears):
                # rows, cols = (feature_list[i], feature_list[i+1])
                rows, cols = (feature_list[i], feature_list[i+1])
                weight_bias_list.append([create_random_weights(rows, cols, density), ([cols], *create_dense_bias(cols))])
                # weight_bias_list.append([])
                # weight_bias_list.append([create_random_weights(rows, cols, 0.8), ([rows], *create_dense_bias(rows))])
    else:
        raise NotImplementedError
    return weight_bias_list

def dump_sparse_csr_fast(tensor_name, sparse_mat, dir_name):
    """
    Fast dump of scipy sparse matrix in CSF (Compressed Sparse Fiber) format.
    Uses numpy vectorized ops instead of Python loops.

    CSF format for 2D matrix:
    - mode_0_seg: [0, num_non_empty_rows]  (2 entries)
    - mode_0_crd: row indices of non-empty rows
    - mode_1_seg: pointer into mode_1_crd for each row (num_non_empty_rows + 1 entries)
    - mode_1_crd: column indices (nnz entries)
    - vals: values (nnz entries)

    Args:
        tensor_name: Name prefix for output files
        sparse_mat: scipy sparse matrix (will be converted to CSR)
        dir_name: Output directory path (should end with /)
    """
    import os
    os.makedirs(dir_name, exist_ok=True)

    # Convert to CSR for efficient row-wise access
    csr = sparse_mat.tocsr()
    rows, cols = csr.shape

    # Find non-empty rows
    row_nnz = np.diff(csr.indptr)
    non_empty_rows = np.where(row_nnz > 0)[0]
    num_non_empty = len(non_empty_rows)

    # Write shape
    with open(dir_name + "tensor_" + tensor_name + "_mode_shape", 'w') as f:
        f.write(f"{rows}\n{cols}\n")

    # Write values
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_vals", csr.data, fmt='%.6f')

    # Mode 0 (rows) - CSF style: seg has [0, num_non_empty_rows]
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_0_seg", [0, num_non_empty], fmt='%d')
    # Row coordinates are the indices of non-empty rows
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_0_crd", non_empty_rows, fmt='%d')

    # Mode 1 (cols) - segment array is cumulative nnz per non-empty row
    # We need to extract only the indptr values for non-empty rows
    mode_1_seg = np.zeros(num_non_empty + 1, dtype=np.int64)
    mode_1_seg[1:] = np.cumsum(row_nnz[non_empty_rows])
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_1_seg", mode_1_seg, fmt='%d')

    # Mode 1 coordinates are the column indices
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_1_crd", csr.indices, fmt='%d')

def dump_dense_1d_fast(tensor_name, data, dir_name):
    """
    Fast dump of 1D array in CSF format (for sparse tensor compatibility).

    For 1D compressed tensors, CSF format is:
    - mode_shape: just the length
    - mode_0_seg: [0, nnz]
    - mode_0_crd: indices of nonzero elements
    - mode_vals: values

    Args:
        tensor_name: Name prefix for output files
        data: 1D numpy array
        dir_name: Output directory path (should end with /)
    """
    import os
    os.makedirs(dir_name, exist_ok=True)

    # Find non-zero elements (for dense bias, all are typically non-zero)
    nonzero_idx = np.where(data != 0)[0]
    if len(nonzero_idx) == 0:
        # If all zeros, still need at least some structure
        nonzero_idx = np.arange(len(data))
        nonzero_vals = data
    else:
        nonzero_vals = data[nonzero_idx]

    # Write shape
    with open(dir_name + "tensor_" + tensor_name + "_mode_shape", 'w') as f:
        f.write(f"{len(data)}\n")

    # Write CSF structure for 1D
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_0_seg", [0, len(nonzero_idx)], fmt='%d')
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_0_crd", nonzero_idx, fmt='%d')
    np.savetxt(dir_name + "tensor_" + tensor_name + "_mode_vals", nonzero_vals, fmt='%.6f')

def dump_data(tensor_name, data, dir_name, shape=None, format="CSF"):
    
    if format == "CSF":
        shape, fiber_tree, vals = data
        with open(dir_name + tensor_name + "_mode_shape", 'w') as f:
            for x in shape:
                f.write(str(x) + "\n")
        with open(dir_name + tensor_name + "_mode_vals", 'w') as f:
            for x in vals:
                f.write(str(x) + "\n")
        for s in range(len(shape)):
            with open(f"{dir_name}{tensor_name}_mode_{s}_crd", 'w') as f:
                for x in fiber_tree[str(s)]["crd"]:
                    f.write(str(x) + "\n")
            with open(f"{dir_name}{tensor_name}_mode_{s}_seg", 'w') as f:
                for x in fiber_tree[str(s)]["seg"]:
                    f.write(str(x) + "\n")
    else:
        with open(dir_name + tensor_name + "_mode_shape", 'w') as f:
            for x in shape:
                f.write(str(x) + "\n")
        with open(dir_name + tensor_name + "_mode_vals", 'w') as f:
            for x in data:
                f.write(str(x) + "\n")
                
class graphsage_sparse:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage/graphsage_sparse/"
        self.linears_per_layer = 2
        self.weights = []
        self.biases = []
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        print(adjacency)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                
        # adj_shape = adjacency[0]
        # adj_fibertree = adjacency[1]
        # adj_vals = adjacency[2]

        # adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        # adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        # pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        # adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # # print(mg.get_matrix())

        # adj = adj_mg.get_matrix().astype(np.float32)
        # x_mat = x.get_matrix().astype(np.float32)

        # adj_x_mul = adj @ x_mat
        # lin = adj_x_mul @ self.weights[1].get_matrix().astype(np.float32) + self.biases[1].get_matrix().astype(np.float32)
        # lin = lin + (x_mat @ self.weights[0].get_matrix().astype(np.float32) + self.biases[0].get_matrix().astype(np.float32))
        # lin = np.maximum(lin, 0.0)
        
        # adj_x_mul2 = adj @ lin
        # neighbor = adj_x_mul2 @ self.weights[3].get_matrix().astype(np.float32) + self.biases[3].get_matrix().astype(np.float32)
        # out = neighbor + (lin @ self.weights[2].get_matrix().astype(np.float32) + self.biases[2].get_matrix().astype(np.float32))
        # out = np.maximum(out, 0.0)
        # # lin = np.maximum(lin, 0.0)
        # # lin = torch.from_numpy(lin)
        # # lin = lin.masked_fill(lin == 0, -1e9)
        # # smax = torch.nn.functional.softmax(lin, dim=-1)
        # # smax = smax.numpy()

        # res = MatrixGenerator("graphsage_gold_result", lin.shape, [0, 1], dump_dir="/tmp/", format="CSF", tensor=out)
        # res.dump_outputs("CSF")
                
class one_layer_graphsage:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/one_layer_graphsage/"
        self.linears_per_layer = 2
        self.weights = []
        self.biases = []

    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)

    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        x.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)

        # Gold Check        
        # adj_shape = adjacency[0]
        # adj_fibertree = adjacency[1]
        # adj_vals = adjacency[2]

        # adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        # adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        # pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        # adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # print(mg.get_matrix())

        # adj = adj_mg.get_matrix()
        # x_mat = x.get_matrix()

        # adj_x_mul = adj @ x_mat
        # lin = adj_x_mul @ self.weights[1].get_matrix() + self.biases[1].get_matrix()
        # lin = lin + (x_mat @ self.weights[0].get_matrix() + self.biases[0].get_matrix())
        # lin = np.maximum(lin, 0.0)
        
        # adj_x_mul2 = adj @ lin
        # neighbor = adj_x_mul2 @ self.weights[3].get_matrix() + self.biases[3].get_matrix()
        # out = neighbor + (lin @ self.weights[2].get_matrix() + self.biases[2].get_matrix())
        # out = np.maximum(out, 0.0)
        # lin = np.maximum(lin, 0.0)
        # lin = torch.from_numpy(lin)
        # lin = lin.masked_fill(lin == 0, -1e9)
        # smax = torch.nn.functional.softmax(lin, dim=-1)
        # smax = smax.numpy()

        # res = MatrixGenerator("gold_result", lin.shape, [0, 1], dump_dir="/tmp/", format="CSF", tensor=lin)
        # res.dump_outputs("CSF")
        
        
class nested_matmuls:
    """
    Data generator for nested_matmuls.mlir that generates sparse matrices
    using KarateClub (or other dataset) graph structure.

    This version properly tracks mode ordering for each tensor to handle
    different dataflow loop orders which may require transposed tensors.
    """
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/kernels/nested_matmuls/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        # Track all tensors with their mode orderings for proper transpose handling
        self.tensors = {}  # tensor_name -> {'mode': mode, 'input_arg': input_arg, 'format': format}
        self.adjmode = []
        self.xmode = []
        self.weightmode = []

    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        # Store all tensor metadata for proper mode ordering
        self.tensors[tensor_name] = {
            'mode': mode,
            'input_arg': input_arg,
            'format': format
        }

        if input_arg == 0:
            self.adjname = tensor_name
            self.adjmode = mode
        elif input_arg == 1:
            self.xname = tensor_name
            self.xmode = mode
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
            self.weightmode = mode

    def generate_data(self, weight_sparsity=None):
        print(f"adjmode: {self.adjmode}")
        print(f"xmode: {self.xmode}")
        print(f"All tensors: {self.tensors}")

        # Third tensor (t2) uses 50% sparsity by default
        sparsity = weight_sparsity if weight_sparsity is not None else 0.5
        dump_dir = os.getcwd() + self.file_path

        # Load dataset
        data, _ = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        if type(data.x) == torch.Tensor:
            features = data.x.numpy()
        else:
            features = data.x

        # Create adjacency matrix from edge_index
        edge_index = data.edge_index.numpy()
        num_nodes = features.shape[0]
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Generate each tensor with proper mode ordering
        # t0 (input_arg=0): sparse adjacency matrix (CSF)
        # t1 (input_arg=1): DENSE matrix stored as dense
        # t2 (input_arg=2): sparse matrix with 50% sparsity (CSF)
        for tensor_name, info in self.tensors.items():
            mode = info['mode']
            input_arg = info['input_arg']

            # Determine if we need to transpose based on mode ordering
            # mode [0,1] = row-major (normal), mode [1,0] = column-major (transposed)
            needs_transpose = (mode == [1, 0])

            if input_arg == 0:
                # First tensor: Sparse adjacency matrix (CSF format)
                tensor_data = adj.T if needs_transpose else adj
                mg = MatrixGenerator(tensor_name, tensor_data.shape, mode,
                                   dump_dir=dump_dir, tensor=tensor_data, format="CSF")
                print(f"  Generated {tensor_name}: mode={mode}, transpose={needs_transpose}, format=CSF (sparse)")
                mg.dump_outputs("CSF")
            elif input_arg == 1:
                # Second tensor: Dense 34x34 matrix (UNC format) - matches paper configuration
                shape = (num_nodes, num_nodes)
                if needs_transpose:
                    shape = (shape[1], shape[0])
                mg = MatrixGenerator(tensor_name, shape, mode,
                                   dump_dir=dump_dir, format="UNC", sparsity=sparsity)
                print(f"  Generated {tensor_name}: mode={mode}, transpose={needs_transpose}, format=UNC (dense {shape[0]}x{shape[1]})")
                mg.dump_outputs("UNC")
            else:
                # Third tensor (t2): Weight matrix 34x16 (num_nodes x hidden_channels) with 50% sparsity
                shape = (num_nodes, self.hidden_channels)
                if needs_transpose:
                    shape = (shape[1], shape[0])
                mg = MatrixGenerator(tensor_name, shape, mode,
                                   dump_dir=dump_dir, format="CSF", sparsity=sparsity)
                print(f"  Generated {tensor_name}: mode={mode}, transpose={needs_transpose}, format=CSF (sparse {shape[0]}x{shape[1]}, {sparsity*100}% sparsity)")
                mg.dump_outputs("CSF")

            self.weights.append(mg)
                
        # adj_shape = adjacency[0]
        # adj_fibertree = adjacency[1]
        # adj_vals = adjacency[2]

        # adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        # adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        # pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        # adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # # print(mg.get_matrix())

        # adj = adj_mg.get_matrix()
        # x_mat = x.get_matrix()

        # adj_x_mul = adj @ x_mat
        # lin = adj_x_mul @ self.weights[1].get_matrix() + self.biases[1].get_matrix()
        # lin = lin + (x_mat @ self.weights[0].get_matrix() + self.biases[0].get_matrix())
        # lin = np.maximum(lin, 0.0)
        
        # # adj_x_mul2 = adj @ lin
        # # neighbor = adj_x_mul2 @ self.weights[3].get_matrix() + self.biases[3].get_matrix()
        # # out = neighbor + (lin @ self.weights[2].get_matrix() + self.biases[2].get_matrix())
        # # out = np.maximum(out, 0.0)
        # # lin = np.maximum(lin, 0.0)
        # # lin = torch.from_numpy(lin)
        # # lin = lin.masked_fill(lin == 0, -1e9)
        # # smax = torch.nn.functional.softmax(lin, dim=-1)
        # # smax = smax.numpy()

        # res = MatrixGenerator("gold_result", lin.shape, [0, 1], dump_dir="/tmp/", format="CSF", tensor=lin)
        # res.dump_outputs("CSF")


class nested_matmuls_synthetic:
    """
    Synthetic data generator for nested_matmuls.mlir that generates
    random sparse matrices matching the dimensions specified in the MLIR file.
    """
    def __init__(self, args):
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.9
        self.file_path = "/data/tests/nested_matmuls/"
        self.tensors = {}  # tensor_name -> (shape, mode, format)

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        """Collect tensor metadata from MLIR parsing."""
        if shape is not None:
            # Convert shape strings to ints
            shape_ints = [int(s) for s in shape]
            self.tensors[tensor_name] = {
                'shape': shape_ints,
                'mode': mode,
                'format': format if format else 'CSF',
                'input_arg': input_arg
            }
            print(f"Collected tensor: {tensor_name}, shape={shape_ints}, mode={mode}, format={format}")

    def generate_data(self, weight_sparsity=None):
        """Generate synthetic sparse matrices for all collected tensors."""
        sparsity = weight_sparsity if weight_sparsity is not None else self.sparsity
        dump_dir = os.getcwd() + self.file_path

        print(f"Generating synthetic data for {len(self.tensors)} tensors with sparsity={sparsity}")
        print(f"Using dump directory - {dump_dir}")

        for tensor_name, info in self.tensors.items():
            shape = info['shape']
            mode = info['mode']
            fmt = info['format']

            # Determine output format based on tensor format
            # Dense tensors use UNC, sparse (compressed) use CSF
            # fmt can be a list like ['compressed', 'compressed'] or ['dense', 'dense'] or just 'dense'
            is_dense = False
            if isinstance(fmt, list):
                is_dense = all(f == 'dense' for f in fmt)
            elif isinstance(fmt, str):
                is_dense = fmt == 'dense'

            if is_dense:
                out_format = 'UNC'
                tensor_sparsity = 0.0  # Dense tensors have no sparsity
            else:
                out_format = 'CSF'
                tensor_sparsity = sparsity

            print(f"  Generating {tensor_name}: shape={shape}, mode={mode}, format={out_format}, sparsity={tensor_sparsity}")

            mg = MatrixGenerator(
                tensor_name,
                shape,
                mode,
                dump_dir=dump_dir,
                format=out_format,
                sparsity=tensor_sparsity
            )
            mg.dump_outputs(out_format)


class one_layer_graphsage2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/one_layer_graphsage2/"
        self.linears_per_layer = 2
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)

        adj_shape = adjacency[0]
        adj_fibertree = adjacency[1]
        adj_vals = adjacency[2]

        adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # print(mg.get_matrix())

        adj = adj_mg.get_matrix()
        x_mat = x.get_matrix()

        adj_x_mul = adj @ x_mat
        lin = adj_x_mul @ self.weights[1].get_matrix() + self.biases[1].get_matrix()
        lin = lin + (x_mat @ self.weights[0].get_matrix() + self.biases[0].get_matrix())
        lin = np.maximum(lin, 0.0)
        
        # adj_x_mul2 = adj @ lin
        # neighbor = adj_x_mul2 @ self.weights[3].get_matrix() + self.biases[3].get_matrix()
        # out = neighbor + (lin @ self.weights[2].get_matrix() + self.biases[2].get_matrix())
        # out = np.maximum(out, 0.0)
        # lin = np.maximum(lin, 0.0)
        # lin = torch.from_numpy(lin)
        # lin = lin.masked_fill(lin == 0, -1e9)
        # smax = torch.nn.functional.softmax(lin, dim=-1)
        # smax = smax.numpy()

        res = MatrixGenerator("gold_result", lin.shape, [0, 1], dump_dir="/tmp/", format="CSF", tensor=lin)
        res.dump_outputs("CSF")
                
class graphsage_adj_x1:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_adj_x1/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.adjname = tensor_name
        elif input_arg == 1:
            self.xname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # weight_names = self.weight
        # bias_names = self.bias
        # linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        # name_count = 0
        # for i in range(len(linear_shapes)):
        #     for _ in range(self.linears_per_layer):
        #         weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        #         bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][0]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        #         weight.dump_outputs("UNC")
        #         bias.dump_outputs("UNC")
                # name_count += 1

class graphsage_linear1_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        # self.data = None
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear1_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        adj_mul = convert_adjacency_matrix(self.data)
        # res = adj_mul.toarray() @ x
        res = x
        print(self.inname)
        print(self.weightname)
        print("SHAPE1: ", res.shape)
        print("SHAPE2: ", (x.shape[1], self.hidden_channels))
        # adj_x_in = MatrixGenerator(self.inname, res.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, tensor=res, format="UNC")
        adj_x_in = MatrixGenerator(self.inname, x.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        weight = MatrixGenerator(self.weightname, (x.shape[1], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        adj_x_in.dump_outputs("UNC")
        weight.dump_outputs("CSF")
        

class graphsage_linear_adds:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        # self.data = None
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear_adds/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        adj_mul = convert_adjacency_matrix(self.data)
        # res = adj_mul.toarray() @ x
        res = x
        print(self.inname)
        print(self.weightname)
        print("SHAPE1: ", res.shape)
        print("SHAPE2: ", (x.shape[1], self.hidden_channels))
        # adj_x_in = MatrixGenerator(self.inname, res.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, tensor=res, format="UNC")
        lin_self = MatrixGenerator(self.inname, x.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        lin_nbor = MatrixGenerator(self.weightname, x.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # weight = MatrixGenerator(self.weightname, (x.shape[1], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        lin_self.dump_outputs("UNC")
        lin_nbor.dump_outputs("UNC")

        
class graphsage_linear1_mul2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        # self.data = None
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear1_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        adj_mul = convert_adjacency_matrix(self.data)
        # res = adj_mul.toarray() @ x
        res = x
        print(self.inname)
        print(self.weightname)
        print("SHAPE1: ", res.shape)
        print("SHAPE2: ", (x.shape[1], self.hidden_channels))
        # adj_x_in = MatrixGenerator(self.inname, res.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, tensor=res, format="UNC")
        adj_x_in = MatrixGenerator(self.inname, x.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        weight = MatrixGenerator(self.weightname, (x.shape[1], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        adj_x_in.dump_outputs("UNC")
        weight.dump_outputs("CSF")
            

class graphsage_linear1_bias:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear1_bias/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias = MatrixGenerator(self.weightname, (x.shape[1],), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        bias.dump_outputs("UNC")
        

class graphsage_relu:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_relu/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        x = self.data.x.numpy()
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # bias = MatrixGenerator(self.weightname, (x.get_shape()[1],), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        # bias.dump_outputs("UNC")


class graphsage_adj_x2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_adj_x2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.adjname = tensor_name
        elif input_arg == 1:
            self.xname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")


class graphsage_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear2_mul/"
        self.linears_per_layer = 1
        np.random.seed(0)
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        print("RETRIEVED DATA")
        print(self.data)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        print(self.data)
        x = self.in_channels
        print("Channels size: ", self.in_channels)
        adj_x_in = MatrixGenerator(self.inname, (self.in_channels, self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        weight = MatrixGenerator(self.weightname, (self.hidden_channels, self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        adj_x_in.dump_outputs("UNC")
        weight.dump_outputs("UNC")
        

class graphsage_linear2_bias:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_linear2_bias/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias = MatrixGenerator(self.weightname, (self.out_channels,), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        bias.dump_outputs("UNC")
        

class graphsage_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/graphsage_unfused/graphsage_softmax/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print("NAME: ", self.inname)
        # print("WEIGHT", self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # bias = MatrixGenerator(self.weightname, (x.shape[1],), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
                
                
# TODO: END GRAPHSAGE HERE
            

class gcn_sparse:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn/gcn_sparse/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xmode = []
        self.adjmode = []
        self.weightmode = []
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xmode = mode
        elif input_arg == 1:
            self.adjname = tensor_name
            self.adjmode = mode
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
            
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, "t11-1,0", self.file_path, adjmode=[1, 0], xmode=self.xmode)
        print(self.adjname)
        print(self.adjmode)
        # print(adjacency.get_shape())
        # exit(0)
        # adjacency.dump_outputs("CSF")
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path, adjmode=[0, 1], xmode=self.xmode)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                
        # adj_shape = adjacency.get_shape()
        # adj_fibertree, adj_vals = adjacency.get_fibertree()

        # adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        # adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        # pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        # adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # # print(mg.get_matrix())

        # adj = adj_mg.get_matrix()
        # adj_x_mul = adj @ x.get_matrix()
        # lin = adj_x_mul @ self.weights[0].get_matrix() + self.biases[0].get_matrix()
        # lin = np.maximum(lin, 0.0)
        
        # adj_x_mul2 = adj @ lin
        # lin = adj_x_mul2 @ self.weights[1].get_matrix() + self.biases[1].get_matrix()
        # # lin = np.maximum(lin, 0.0)
        # lin = torch.from_numpy(lin)
        # lin = lin.masked_fill(lin == 0, -1e9)
        # smax = torch.nn.functional.softmax(lin, dim=-1)
        # smax = smax.numpy()

        # res = MatrixGenerator("gold_result", smax.shape, [0, 1], dump_dir="/tmp/", format="UNC", tensor=smax)
        # res.dump_outputs("UNC")
                

class one_layer_gcn:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/one_layer_gcn/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xmode = []
        self.adjmode = []
        self.weightmode = []
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xmode = mode
        elif input_arg == 1:
            self.adjname = tensor_name
            self.adjmode = mode
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
            self.weightmode = mode
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path, self.adjmode, self.xmode)

        print("X MODE: ", self.xmode)
        print("ADJ MODE: ", self.adjmode)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        print("MODE")
        print(self.xmode)
        print(self.adjmode)
        print(self.weightmode)
        # exit(0)
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], mode_ordering=self.weightmode, dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)

        
        # adj_shape = adjacency[0]
        # adj_fibertree = adjacency[1]
        # adj_vals = adjacency[2]

        # adj_crds = [adj_fibertree['0']['crd'], adj_fibertree['1']['crd']]
        # adj_segs = [adj_fibertree['0']['seg'], adj_fibertree['1']['seg']]

        # pt_lst = get_point_list(adj_crds, adj_segs, val_arr=adj_vals)
        # adj_mg = create_matrix_from_point_list("adj", pt_lst, adj_shape)

        # # print(mg.get_matrix())

        # adj_x_mul = adj_mg.get_matrix() @ x.get_matrix()
        # lin = adj_x_mul @ self.weights[0].get_matrix() + self.biases[0].get_matrix()
        # lin = np.maximum(lin, 0.0)
        # print(lin)
        # res = MatrixGenerator("gold_result", lin.shape, [0, 1], dump_dir="/tmp/", format="CSF", tensor=lin)
        # res.dump_outputs("CSF")
        


class adj_linear1_relu_adj2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear1_relu_adj2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                

class adj_linear1_relu_adj2_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear1_relu_adj2_linear2_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        weight = MatrixGenerator(weight_names[1], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                

class adj_linear1_relu_adj2_linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear1_relu_adj2_linear2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        name_count = 0
        # weight = MatrixGenerator(weight_names[1], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        # weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1

        
class adj_linear1_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear1_mul/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                # bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                # bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                # self.biases.append(bias)
                
class adj_linear1:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear1/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                # self.biases.append(bias)
                
                
class linear1:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {x.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                

class linear1_relu:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_relu/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {x.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                

class linear1_relu_adj:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_relu_adj/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {x.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                

class linear1_relu_adj_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_relu_adj_linear2_mul/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {x.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        weight = MatrixGenerator(weight_names[1], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                

class linear1_relu_adj_linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_relu_adj_linear2/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {x.get_shape()}")
        x.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        name_count = 0
        # weight = MatrixGenerator(weight_names[1], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        # weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                
                
class linear1_bias_relu:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_bias_relu/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # print(f"shape of features {x.get_shape()}")
        # x.dump_outputs("UNC")
        # print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        x_data = self.data.x.numpy()
        print(x_data.shape)
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        x = MatrixGenerator(self.xname, [x_data.shape[0], self.hidden_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print("intermediate:", x.shape)
        x.dump_outputs("UNC")
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                # weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                # weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                # print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                self.biases.append(bias)
                
                

class linear1_bias_relu_adj2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_bias_relu_adj2/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # print(f"shape of features {x.get_shape()}")
        # x.dump_outputs("UNC")
        # print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x_data = self.data.x.numpy()
        print(x_data.shape)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        x = MatrixGenerator(self.xname, [x_data.shape[0], self.hidden_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        x.dump_outputs("UNC")
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                # weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                # weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                # print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                self.biases.append(bias)
                

class linear1_bias_relu_adj2_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_bias_relu_adj2_linear2_mul/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # print(f"shape of features {x.get_shape()}")
        # x.dump_outputs("UNC")
        # print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x_data = self.data.x.numpy()
        print(x_data.shape)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        x = MatrixGenerator(self.xname, [x_data.shape[0], self.hidden_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        x.dump_outputs("UNC")
        name_count = 0
        print(weight_names)
        weight = MatrixGenerator(weight_names[0], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                print(bias_names[name_count])
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                bias.dump_outputs("UNC")
                # print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                self.biases.append(bias)
                
                
class linear1_bias_relu_adj2_linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_bias_relu_adj2_linear2/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # print(f"shape of features {x.get_shape()}")
        # x.dump_outputs("UNC")
        # print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x_data = self.data.x.numpy()
        print(x_data.shape)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        x = MatrixGenerator(self.xname, [x_data.shape[0], self.hidden_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        x.dump_outputs("UNC")
        name_count = 0
        print(weight_names)
        weight = MatrixGenerator(weight_names[0], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        bias = MatrixGenerator(bias_names[1], [self.out_channels], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias.dump_outputs("UNC")
        weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                print(bias_names[name_count])
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                bias.dump_outputs("UNC")
                # print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                self.biases.append(bias)
                
                
class linear1_bias_relu_adj2_linear2_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear1_bias_relu_adj2_linear2_softmax/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # print(f"shape of features {x.get_shape()}")
        # x.dump_outputs("UNC")
        # print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x_data = self.data.x.numpy()
        print(x_data.shape)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        x = MatrixGenerator(self.xname, [x_data.shape[0], self.hidden_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        x.dump_outputs("UNC")
        name_count = 0
        print(weight_names)
        weight = MatrixGenerator(weight_names[0], [self.hidden_channels, self.out_channels], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        bias = MatrixGenerator(bias_names[1], [self.out_channels], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias.dump_outputs("UNC")
        weight.dump_outputs("CSF")
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                print(bias_names[name_count])
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                bias.dump_outputs("UNC")
                # print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                self.biases.append(bias)
        
                
                
class gcn_fused_across:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_fused_across/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        x.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.in_channels, self.hidden_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                
class adj_linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        print(xinter.get_shape())
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        print(linear_shapes)
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
 
                

class one_layer_gcn2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/one_layer_gcn2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(xinter.get_shape())
        xinter.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF", sparsity=0.5)
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                print("Bias: ", linear_shapes[i][1])
                name_count += 1
                

class relu_adj2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/relu_adj2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        # x.dump_outputs("UNC")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        print(self.xname)
        print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        adjacency.dump_outputs("CSF")
        
    

class gcn_adj_x1:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_adj_x1/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.adjname = tensor_name
        elif input_arg == 1:
            self.xname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        x.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        # weight_names = self.weight
        # bias_names = self.bias
        # linear_shapes = [(self.in_channels, self.hidden_channels), (self.hidden_channels, self.out_channels)]
        # name_count = 0
        # for i in range(len(linear_shapes)):
        #     for _ in range(self.linears_per_layer):
        #         weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        #         bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][0]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        #         weight.dump_outputs("UNC")
        #         bias.dump_outputs("UNC")
                # name_count += 1

class gcn_linear1_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        # self.data = None
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_linear1_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        adj_mul = convert_adjacency_matrix(self.data)
        # res = adj_mul.toarray() @ x
        res = x
        print(self.inname)
        print(self.weightname)
        print("SHAPE1: ", res.shape)
        print("SHAPE2: ", (x.shape[1], self.hidden_channels))
        # adj_x_in = MatrixGenerator(self.inname, res.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, tensor=res, format="UNC")
        adj_x_in = MatrixGenerator(self.inname, x.shape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        weight = MatrixGenerator(self.weightname, (x.shape[1], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        adj_x_in.dump_outputs("UNC")
        weight.dump_outputs("CSF")
            

class gcn_linear1_bias:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_linear1_bias/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias = MatrixGenerator(self.weightname, (self.hidden_channels,), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        bias.dump_outputs("UNC")
        

class gcn_relu:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_relu/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        x = self.data.x.numpy()
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC", sparsity=.5)
        # linear_out = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC", sparsity=.5)
        # bias = MatrixGenerator(self.weightname, (x.get_shape()[1],), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        # bias.dump_outputs("UNC")


class gcn_adj_x2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_adj_x2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.adjname = tensor_name
        elif input_arg == 1:
            self.xname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        # x.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(linear_out.get_shape())
        linear_out.dump_outputs("UNC")


class adj_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/adj_linear2_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        # adjacency.dump_outputs("CSF")
        dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")

        weight_names = self.weight
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                # bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                # bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                

class relu_adj2_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/relu_adj2_linear2_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print("adjname", self.adjname)
        print("xname", self.xname)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        # linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # linear_out.dump_outputs("UNC")
        # xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # xinter.dump_outputs("UNC")

        print("weight", self.weight)

        weight_names = self.weight
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                # bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                # bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                
                
class relu_adj2_linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/relu_adj2_linear2/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print("adjname", self.adjname)
        print("xname", self.xname)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        # linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # linear_out.dump_outputs("UNC")
        # xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # xinter.dump_outputs("UNC")

        print("weight", self.weight)

        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)
                

class relu_adj2_linear2_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/relu_adj2_linear2_softmax/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
        elif input_arg == 1:
            pass
        elif input_arg == 2:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        print("adjname", self.adjname)
        print("xname", self.xname)
        print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency.get_shape()}")
        xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        adjacency.dump_outputs("CSF")
        # linear_out = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # linear_out.dump_outputs("UNC")
        # xinter = MatrixGenerator(self.xname, (x.get_shape()[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # xinter.dump_outputs("UNC")

        print("weight", self.weight)

        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                # self.weights.append(weight)


class linear2:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear2/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {self.in_channels}")
        # x.dump_outputs("UNC")
        x = self.data.x.numpy()
        xinter = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)
                

class linear2_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear2_softmax/"
        self.linears_per_layer = 1
        self.weights = []
        self.biases = []
        self.xname = ""
        self.adjname = ""
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.xname = tensor_name
            self.xshape = shape
        elif input_arg == 1:
            self.adjname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, self.xname, self.adjname, self.file_path)
        # x = MatrixGenerator(self.xname, xshape, [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(f"shape of features {self.in_channels}")
        # x.dump_outputs("UNC")
        x = self.data.x.numpy()
        xinter = MatrixGenerator(self.xname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        xinter.dump_outputs("UNC")
        print(self.xname)
        # print(self.adjname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # adjacency.dump_outputs("CSF")
        weight_names = self.weight
        bias_names = self.bias
        linear_shapes = [(self.hidden_channels, self.out_channels)] #, (self.hidden_channels, self.out_channels)]
        name_count = 0
        for i in range(len(linear_shapes)):
            for _ in range(self.linears_per_layer):
                weight = MatrixGenerator(weight_names[name_count], linear_shapes[i], [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
                bias = MatrixGenerator(bias_names[name_count], [linear_shapes[i][1]], [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
                weight.dump_outputs("CSF")
                bias.dump_outputs("UNC")
                print(weight_names[name_count])
                # print(bias_names[name_count])
                print("Weight: ", linear_shapes[i])
                # print("Bias: ", linear_shapes[i][1])
                name_count += 1
                self.weights.append(weight)
                self.biases.append(bias)


class gcn_linear2_mul:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_linear2_mul/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        print("RETRIEVED DATA")
        print(self.data)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        print(self.data)
        x = self.data.x.numpy()
        # xinter = MatrixGenerator(self.inname, (self.in_channels, self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        # xinter.dump_outputs("UNC")
        # print(x.shape)
        # print("Channels size: ", self.in_channels)
        adj_x_in = MatrixGenerator(self.inname, (x.shape[0], self.hidden_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        weight = MatrixGenerator(self.weightname, (self.hidden_channels, self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="CSF")
        print("COUNT: ", np.count_nonzero(weight.get_matrix()))
        adj_x_in.dump_outputs("UNC")
        weight.dump_outputs("CSF")
        

class gcn_linear2_bias:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_linear2_bias/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        print(tensor_name)
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.biasname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.biasname = tensor_name
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print(self.inname)
        print(self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        x = self.data.x.numpy()
        print(x.shape)
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias = MatrixGenerator(self.biasname, (self.out_channels,), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        bias.dump_outputs("UNC")
        
        
class linear2_bias_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_fusion_ablation/linear2_bias_softmax/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.biasname = tensor_name
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        print(self.inname)
        # print(self.data.shape)
        x = self.data.x.numpy()
        linear_out = MatrixGenerator(self.inname, (x.shape[0], self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        bias = MatrixGenerator(self.biasname, (self.out_channels,), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")

        print(linear_out.get_shape())
        linear_out.dump_outputs("UNC")
        bias.dump_outputs("UNC")
        

class gcn_softmax:
    def __init__(self, args):
        dataset_name = args.inDataset
        name = args.inData
        self.input_args = 2
        self.inname = ""
        self.weightname = ""
        self.dataset_name = dataset_name
        self.name = name
        self._prep_dataset()
        self.hidden_channels=16
        self.bias = []
        self.weight = []
        self.sh1 = []
        self.sh2 = []
        self.file_path = "/data/gcn_unfused/gcn_softmax/"
        self.linears_per_layer = 1
    
    def _prep_dataset(self):
        self.root_dir = os.environ["DATA_PATH"]
        self.data, num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        self.in_channels = self.data.num_features
        self.out_channels = num_classes

    def gen_data(self, tensor_name, input_args, mode_info, shape=None):        
        if input_args < self.input_args:
            x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name)
            if input_args == 0:
                dump_data(tensor_name, x, os.getcwd() + self.file_path)
                return x[0]
            else:
                dump_data(tensor_name, adjacency, os.getcwd() + self.file_path)
                return adjacency[0]
        else:
            raise NotImplementedError 
    
    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        if input_arg == 0:
            self.inname = tensor_name
        elif input_arg == 1:
            self.weightname = tensor_name
        elif len(mode) == 1:
            self.bias.append(tensor_name)
            self.sh1.append(shape)
        elif len(mode) == 2:
            self.weight.append(tensor_name)
            self.sh2.append(shape)
    
    def generate_data(self):
        # x, adjacency = data_gen(self.dataset_name, self.root_dir, self.name, path=self.file_path)
        # print(f"shape of features {x.get_shape()}, shape of adjacency {adjacency[0]}")
        # x.dump_outputs("UNC")
        print("NAME: ", self.inname)
        # print("WEIGHT", self.weightname)
        # dump_data("tensor_" + self.adjname, adjacency, os.getcwd() + self.file_path)
        # x = self.data.x.numpy()
        print(self.data.x.numpy().shape)
        linear_out = MatrixGenerator(self.inname, (self.data.x.numpy().shape[0], self.out_channels), [0, 1], dump_dir=os.getcwd() + self.file_path, format="UNC")
        print(linear_out.get_shape())
        # bias = MatrixGenerator(self.weightname, (x.shape[1],), [0], dump_dir=os.getcwd() + self.file_path, format="UNC")
        linear_out.dump_outputs("UNC")
        # bias.dump_outputs("UNC")


class multihead_attention:
    def __init__(self, args):
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.kshape = None
        self.vshape = None
        self.qmode = None
        self.kmode = None
        self.vmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = args.block
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3/multihead_attention"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(tensor_name)
        if input_arg == 0:
            self.qname = tensor_name
            self.qshape = shape
            self.qmode = mode
        elif input_arg == 1:
            self.kname = tensor_name
            self.kshape = shape
            self.kmode = mode
        elif input_arg == 2:
            self.vname = tensor_name
            self.vshape = shape
            self.vmode = mode
        else:
            self.maskname = tensor_name
            self.maskshape = shape
            self.maskmode = mode

    def get_mask(self):
        shape = self.maskshape
        maskTens = np.zeros(shape)

        num_rand_blocks = 2
        np.random.seed(0)
        for i in range(shape[0]):
            if i % self.block_size == 0:
                random_ind = np.random.choice(shape[0] // self.block_size, num_rand_blocks, replace=False)
                for ind in random_ind:
                    ind = ind * self.block_size
                    if ind % self.block_size == 0:
                        for m in range(self.block_size):
                            for n in range(self.block_size):
                                maskTens[i + m, ind + n] = 1
            for j in range(shape[1]):
                if i == j or i == j - self.block_size  or i == j + self.block_size or i < 2 * self.block_size or j < 2 * self.block_size:
                    if i % self.block_size == 0 and j % self.block_size == 0:
                        print(i, j)
                        for m in range(self.block_size):
                            for n in range(self.block_size):
                                maskTens[i + m][j +n] = 1
        return maskTens
    def generate_data(self):
        # self.kmode = [0, 1, 2, 3]
        mat_q = MatrixGenerator(self.qname, self.qshape,self.qmode, dump_dir=os.getcwd() + self.file_path, format="UNC")
        mat_q.dump_outputs(format='UNC')
        mat_k = MatrixGenerator(self.kname, self.kshape,self.kmode, dump_dir=os.getcwd() + self.file_path, format="UNC")
        mat_v = MatrixGenerator(self.vname, self.vshape,self.vmode, dump_dir=os.getcwd() + self.file_path, format="UNC")
        mat_v.dump_outputs(format='UNC')
        mat_mask = MatrixGenerator(self.maskname, self.maskshape,self.maskmode, dump_dir=os.getcwd() + self.file_path, format="CSF", tensor=self.get_mask())
        mat_mask.dump_outputs(format='CSF')

        q = torch.from_numpy(mat_q.get_matrix()).to(torch.float)
        k = torch.from_numpy(mat_k.get_matrix()).to(torch.float).transpose(-1, -2)
        mat_k = MatrixGenerator(self.kname, self.kshape,self.kmode, dump_dir=os.getcwd() + self.file_path, format="UNC", tensor=k.numpy())
        print("Q Shape: ", q.shape)
        print(k.shape)
        # print(v.shape)
        # k = torch.reshape(k, (1, 12, 64, 128))
        mask = torch.from_numpy(mat_mask.get_matrix()).to(torch.float)
        v = torch.from_numpy(mat_v.get_matrix()).to(torch.float)

        # Gold calculation commented out due to shape mismatch with K tensor transpose
        # qk = torch.einsum('iljm, ilmn->iljn', q, k)
        # qk = qk * (1 / math.sqrt(q.shape[3]))
        # qk = torch.mul(qk, torch.unsqueeze(torch.unsqueeze(mask.to(torch.float), 0), 1))
        # qk = qk.masked_fill(qk == 0, -1e9).to(torch.float)
        # soft = torch.nn.functional.softmax(qk, dim=-1)
        # qkv = torch.einsum('iljn, ilnm->iljm', soft, v)
        # qkv = MatrixGenerator("mha_out", qkv.shape,[0, 1, 2, 3], dump_dir="/tmp", format="CSF", tensor=qkv)
        # qkv.dump_outputs(format='CSF')
        mat_k.dump_outputs(format='UNC')


class multihead_attention_blocked:
    """Block sparse version of multihead_attention for true block sparse simulation.

    This class generates data in BCSR format where each coordinate maps to an NxN
    dense block of values. Used with --trueblock flag.
    """
    def __init__(self, args):
        np.random.seed(0)
        random.seed(0)
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.kshape = None
        self.vshape = None
        self.qmode = None
        self.kmode = None
        self.vmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = 1  # For CSF coordinate traversal
        self.actual_block_size = args.block  # True block size (16, 32, 64)
        self.logical_qshape = None
        self.logical_kshape = None
        self.logical_vshape = None
        self.logical_maskshape = None
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3/multihead_attention_blocked"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"collect_names: {tensor_name}, input_arg={input_arg}, shape={shape}")

        # Detect tensor type by shape: mask is 2D, Q/K/V are 4D
        is_mask = len(shape) == 2

        if is_mask:
            # This is the attention mask (2D tensor)
            self.maskname = tensor_name
            self.logical_maskshape = shape.copy()
            self.maskshape = shape.copy()
            self.maskshape[-2] = self.maskshape[-2] // self.actual_block_size
            self.maskshape[-1] = self.maskshape[-1] // self.actual_block_size
            self.maskshape = [int(x) for x in self.maskshape]
            self.maskmode = mode
        elif input_arg == 0:
            self.qname = tensor_name
            self.logical_qshape = shape.copy()
            self.qshape = shape.copy()
            # Scale down last 2 dimensions for block sparse coordinate space
            self.qshape[-2] = self.qshape[-2] // self.actual_block_size
            self.qshape[-1] = self.qshape[-1] // self.actual_block_size
            self.qshape = [int(x) for x in self.qshape]
            self.qmode = mode
        elif input_arg == 1:
            self.kname = tensor_name
            self.logical_kshape = shape.copy()
            self.kshape = shape.copy()
            self.kshape[-2] = self.kshape[-2] // self.actual_block_size
            self.kshape[-1] = self.kshape[-1] // self.actual_block_size
            self.kshape = [int(x) for x in self.kshape]
            self.kmode = mode
        else:
            # Remaining 4D tensor is V
            self.vname = tensor_name
            self.logical_vshape = shape.copy()
            self.vshape = shape.copy()
            self.vshape[-2] = self.vshape[-2] // self.actual_block_size
            self.vshape[-1] = self.vshape[-1] // self.actual_block_size
            self.vshape = [int(x) for x in self.vshape]
            self.vmode = mode

    def get_mask(self):
        """Generate BigBird-style sparse attention mask at block level."""
        shape = self.maskshape
        maskTens = np.zeros(shape)
        num_rand_blocks = 2
        np.random.seed(0)

        for i in range(shape[0]):
            # Random blocks
            random_ind = np.random.choice(shape[0], num_rand_blocks, replace=False)
            for ind in random_ind:
                maskTens[i, ind] = 1
            # Diagonal and nearby blocks
            for j in range(shape[1]):
                if i == j or i == j - 1 or i == j + 1 or i < 2 or j < 2:
                    maskTens[i, j] = 1
        return maskTens

    def generate_data(self, weight_sparsity=0.8):
        bs = self.actual_block_size

        # Reconstruct logical shapes from scaled shapes
        self.logical_qshape = []
        self.logical_kshape = []
        self.logical_vshape = []
        for i, s in enumerate(self.qshape):
            if i >= len(self.qshape) - 2:
                self.logical_qshape.append(s * bs)
            else:
                self.logical_qshape.append(s)
        for i, s in enumerate(self.kshape):
            if i >= len(self.kshape) - 2:
                self.logical_kshape.append(s * bs)
            else:
                self.logical_kshape.append(s)
        for i, s in enumerate(self.vshape):
            if i >= len(self.vshape) - 2:
                self.logical_vshape.append(s * bs)
            else:
                self.logical_vshape.append(s)

        # Generate Q with block_size parameter
        mat_q = MatrixGenerator(self.qname, self.logical_qshape, self.qmode,
                               dump_dir=os.getcwd() + self.file_path, format="UNC",
                               sparsity=0.01, value_cap=10, block_size=bs)
        mat_q.dump_outputs(format='UNC')

        # Generate K with block_size parameter
        mat_k = MatrixGenerator(self.kname, self.logical_kshape, self.kmode,
                               dump_dir=os.getcwd() + self.file_path, format="UNC",
                               sparsity=0.01, value_cap=10, block_size=bs)

        q = torch.from_numpy(mat_q.get_matrix()).to(torch.float)
        k = torch.from_numpy(mat_k.get_matrix()).to(torch.float)

        # Block-wise K transpose for matrix multiply
        transposedK = torch.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for m in range(0, k.shape[2], bs):
                    for n in range(0, k.shape[3], bs):
                        tile = k[i, j, m:m+bs, n:n+bs]
                        transposed_tile = tile.transpose(-1, -2)
                        transposedK[i, j, m:m+bs, n:n+bs] = transposed_tile

        mat_k = MatrixGenerator(self.kname, self.logical_kshape, self.kmode,
                               dump_dir=os.getcwd() + self.file_path, format="UNC",
                               tensor=transposedK.numpy(), block_size=bs)
        mat_k.dump_outputs(format='UNC')

        # Generate V with block_size parameter
        mat_v = MatrixGenerator(self.vname, self.logical_vshape, self.vmode,
                               dump_dir=os.getcwd() + self.file_path, format="UNC",
                               sparsity=0.01, value_cap=10, block_size=bs)
        mat_v.dump_outputs(format='UNC')

        # Generate mask in CSF format at block level
        if self.maskname:
            mask_logical_shape = [s * bs if i >= len(self.maskshape) - 2 else s
                                  for i, s in enumerate(self.maskshape)]
            mat_mask = MatrixGenerator(self.maskname, mask_logical_shape, self.maskmode,
                                      dump_dir=os.getcwd() + self.file_path, format="CSF",
                                      tensor=self.get_mask(), block_size=bs)
            mat_mask.dump_outputs(format='CSF')

        print(f"Q Shape: {q.shape}")
        print(f"K Shape (transposed blocks): {transposedK.shape}")


class mhaQK_mask:
    def __init__(self, args):
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.kshape = None
        self.vshape = None
        self.qmode = None
        self.kmode = None
        self.vmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = args.block
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3_unfused/mhaQK_mask"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(tensor_name)
        # exit(0)
        if input_arg == 0:
            self.qname = tensor_name
            self.qshape = shape
            self.qmode = mode
        elif input_arg == 1:
            self.maskname = tensor_name
            self.maskshape = shape
            self.maskmode = mode

    def get_mask(self):
        shape = self.maskshape
        maskTens = np.zeros(shape)

        num_rand_blocks = 2
        np.random.seed(0)
        for i in range(shape[0]):
            if i % self.block_size == 0:
                random_ind = np.random.choice(shape[0], num_rand_blocks, replace=False)
                for ind in random_ind:
                    if ind % self.block_size == 0:
                        for m in range(self.block_size):
                            for n in range(self.block_size):
                                maskTens[i + m, ind + n] = 1
            for j in range(shape[1]):
                if i == j or i == j - self.block_size  or i == j + self.block_size or i < 2 * self.block_size or j < 2 * self.block_size:
                    if i % self.block_size == 0 and j % self.block_size == 0:
                        for m in range(self.block_size):
                            for n in range(self.block_size):
                                maskTens[i + m][j +n] = 1
        print(maskTens.shape)
        return maskTens
    def generate_data(self):
        mat_q = MatrixGenerator(self.qname, self.qshape,self.qmode, dump_dir=os.getcwd() + self.file_path, format="UNC")
        mat_q.dump_outputs(format='UNC')
        mat_mask = MatrixGenerator(self.maskname, self.maskshape,self.maskmode, dump_dir=os.getcwd() + self.file_path, format="CSF", tensor=self.get_mask())
        mat_mask.dump_outputs(format='CSF')
        

class mhaQK_mul2:
    def __init__(self, args):
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.qmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = 64
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3_unfused/mhaQK_mul2"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        if input_arg == 0:
            self.maskname = tensor_name
            self.maskshape = shape
            self.maskmode = mode
        elif input_arg == 1:
            self.qname = tensor_name
            self.qshape = shape
            self.qmode = mode

    def get_mask(self):
        shape = self.maskshape
        maskTens = np.zeros(shape)
        print("SHAPE:", shape)

        num_rand_blocks = 2
        np.random.seed(0)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for i in range(shape[-2]):
                    if i % self.block_size == 0:
                        random_ind = np.random.choice(shape[-2], num_rand_blocks, replace=False)
                        for ind in random_ind:
                            if ind % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m, ind + n] = 1
                    for j in range(shape[-1]):
                        if i == j or i == j - self.block_size  or i == j + self.block_size or i < 2 * self.block_size or j < 2 * self.block_size:
                            if i % self.block_size == 0 and j % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m][j +n] = 1
        return maskTens
    def generate_data(self):
        mat_q = MatrixGenerator(self.qname, self.qshape,self.qmode, dump_dir=os.getcwd() + self.file_path, format="UNC")
        mat_q.dump_outputs(format='UNC')
        mat_mask = MatrixGenerator(self.maskname, self.maskshape,self.maskmode, dump_dir=os.getcwd() + self.file_path, format="CSF", tensor=self.get_mask())
        mat_mask.dump_outputs(format='CSF')
        

class mhaQK_softmax:
    def __init__(self, args):
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.qmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = 64
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3_unfused/mhaQK_softmax"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        if input_arg == 0:
            self.maskname = tensor_name
            self.maskshape = shape
            self.maskmode = mode

    def get_mask(self):
        shape = self.maskshape
        maskTens = np.zeros(shape)
        print("SHAPE:", shape)

        num_rand_blocks = 3
        np.random.seed(0)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for i in range(shape[2]):
                    if i % self.block_size == 0:
                        random_ind = np.random.choice(shape[2], num_rand_blocks, replace=False)
                        for ind in random_ind:
                            if ind % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m, ind + n] = 1
                    for j in range(shape[3]):
                        if i == j or i == j - self.block_size  or i == j + self.block_size or i < 2 * self.block_size or j < 2 * self.block_size:
                            if i % self.block_size == 0 and j % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m][j +n] = 1
        print(np.count_nonzero(maskTens) / float(np.prod(shape)))
        print(np.count_nonzero(maskTens))
        return maskTens
    def generate_data(self):
        mat_mask = MatrixGenerator(self.maskname, self.maskshape,self.maskmode, dump_dir=os.getcwd() + self.file_path, format="CSF", tensor=self.get_mask())
        mat_mask.dump_outputs(format='CSF')
        

class mhaQK_scale:
    def __init__(self, args):
        self.args = args
        self.maskname = None
        self.qname = None
        self.qshape = None
        self.qmode = None
        self.maskmode = None
        self.maskshape = None
        self.block_size = 64
        self.vname = None
        self.kname = None
        self.input_args = 3
        self.file_path = "/data/gpt-3_unfused/mhaQK_scale"

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        if input_arg == 0:
            self.maskname = tensor_name
            self.maskshape = shape
            self.maskmode = mode

    def get_mask(self):
        shape = self.maskshape
        maskTens = np.zeros(shape)
        print("SHAPE:", shape)

        num_rand_blocks = 2
        np.random.seed(0)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for i in range(shape[-2]):
                    if i % self.block_size == 0:
                        random_ind = np.random.choice(shape[-2], num_rand_blocks, replace=False)
                        for ind in random_ind:
                            if ind % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m, ind + n] = 1
                    for j in range(shape[-1]):
                        if i == j or i == j - self.block_size  or i == j + self.block_size or i < 2 * self.block_size or j < 2 * self.block_size:
                            if i % self.block_size == 0 and j % self.block_size == 0:
                                for m in range(self.block_size):
                                    for n in range(self.block_size):
                                        maskTens[x][y][i + m][j +n] = 1
        return maskTens
    def generate_data(self):
        mat_mask = MatrixGenerator(self.maskname, self.maskshape,self.maskmode, dump_dir=os.getcwd() + self.file_path, format="CSF", tensor=self.get_mask())
        mat_mask.dump_outputs(format='CSF')


class autoencoder_synthetic:
    """Data generator for autoencoder with synthetic data"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 5
        self.file_path = "/data/autoencoder"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.5

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"autoencoder_synthetic collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        self.tensors[tensor_name] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format,
            'is_input': (input_arg == 0)
        }

    def generate_data(self):
        print(f"Generating autoencoder synthetic data with sparsity={self.sparsity}")
        dump_dir = os.getcwd() + self.file_path + "/"
        os.makedirs(dump_dir, exist_ok=True)

        for tensor_name, tensor_info in self.tensors.items():
            shape = tensor_info['shape']
            mode = tensor_info['mode']
            is_input = tensor_info.get('is_input', False)

            # Generate weight matrices (2D) with 90% sparsity using fast scipy dump
            if len(shape) == 2:
                density = 0.1  # 90% sparse
                sparse_mat = sp.random(shape[0], shape[1], density=density, format='csr', dtype=np.float32)
                sparse_mat.data *= 0.01  # Scale weights
                print(f"Generated weight {tensor_name} shape {shape}, density={density:.3f}, nnz={sparse_mat.nnz}")
                dump_sparse_csr_fast(tensor_name, sparse_mat, dump_dir)
            elif len(shape) == 1:
                if is_input:
                    # Input vector - generate with some sparsity
                    density = 1.0 - self.sparsity
                    nnz = int(shape[0] * density)
                    indices = np.random.choice(shape[0], nnz, replace=False)
                    values = np.random.randn(nnz).astype(np.float32)
                    sparse_vec = sp.csr_matrix((values, (np.zeros(nnz, dtype=int), indices)), shape=(1, shape[0]))
                    print(f"Generated input {tensor_name} shape {shape}, density={density:.3f}, nnz={nnz}")
                    dump_sparse_csr_fast(tensor_name, sparse_vec, dump_dir)
                else:
                    # Bias vectors - generate random and dump fast
                    bias_data = np.random.randn(shape[0]).astype(np.float32) * 0.01
                    print(f"Generated bias {tensor_name} with shape {shape}")
                    dump_dense_1d_fast(tensor_name, bias_data, dump_dir)


class autoencoder_batched:
    """Data generator for batched autoencoder operations"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 4  # Support up to 4 input tensors
        self.file_path = "/data/autoencoder_batched"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.0

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"autoencoder_batched collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        self.tensors[input_arg] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format
        }

    def generate_data(self):
        print(f"Generating autoencoder batched data with sparsity={self.sparsity}")

        for arg_idx, tensor_info in self.tensors.items():
            tensor_name = tensor_info['name']
            shape = tensor_info['shape']
            mode = tensor_info['mode']

            # Generate random tensor
            tensor_data = np.random.randn(*shape).astype(np.float32)
            if self.sparsity > 0:
                mask = np.random.random(shape) > self.sparsity
                tensor_data = tensor_data * mask

            mat = MatrixGenerator(tensor_name, shape, mode,
                                  dump_dir=os.getcwd() + self.file_path, format="UNC", tensor=tensor_data)
            mat.dump_outputs(format='UNC')
            print(f"Generated tensor {tensor_name} with shape {shape}")


class sae_synthetic:
    """Data generator for Sparse Autoencoder (SAE) with synthetic data"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 3
        self.file_path = "/data/sae"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.5

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"sae_synthetic collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        self.tensors[input_arg] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format
        }

    def generate_data(self):
        print(f"Generating SAE synthetic data with sparsity={self.sparsity}")

        for arg_idx, tensor_info in self.tensors.items():
            tensor_name = tensor_info['name']
            shape = tensor_info['shape']
            mode = tensor_info['mode']

            # Generate random tensor with sparsity for activations
            tensor_data = np.random.randn(*shape).astype(np.float32)

            # Apply ReLU-like sparsity (zero out negative values and some positive)
            if self.sparsity > 0 and 'activation' in tensor_name.lower():
                tensor_data = np.maximum(tensor_data, 0)  # ReLU
                mask = np.random.random(shape) > self.sparsity
                tensor_data = tensor_data * mask

            mat = MatrixGenerator(tensor_name, shape, mode,
                                  dump_dir=os.getcwd() + self.file_path, format="UNC", tensor=tensor_data)
            mat.dump_outputs(format='UNC')
            print(f"Generated SAE tensor {tensor_name} with shape {shape}")


class autoencoder_imagenet_batched:
    """Data generator for ImageNet batched autoencoder: batch=32, 50176 -> 256 -> 50176"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 5  # All tensors including constants
        self.file_path = "/data/autoencoder_imagenet_batched"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.5

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"autoencoder_imagenet_batched collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        # Use tensor_name as key since arg can be empty for constants
        self.tensors[tensor_name] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format,
            'is_input': (input_arg == 0)  # Track if it's the actual input
        }

    def generate_data(self, weight_sparsity=None):
        # Use weight_sparsity if provided, otherwise use self.sparsity
        sparsity = weight_sparsity if weight_sparsity is not None else self.sparsity
        print(f"Generating ImageNet batched autoencoder data with sparsity={sparsity}")
        dump_dir = os.getcwd() + self.file_path + "/"
        os.makedirs(dump_dir, exist_ok=True)

        for tensor_name, tensor_info in self.tensors.items():
            shape = tensor_info['shape']
            is_input = tensor_info.get('is_input', False)

            # Generate input tensor as dense (for standalone encoder/decoder ops)
            if is_input:
                if len(shape) == 2:
                    # Generate dense input batch (e.g., 32x50176 for ImageNet)
                    input_data = np.random.randn(shape[0], shape[1]).astype(np.float32) * 0.1
                    # Dump as dense CSR (all values)
                    dense_as_csr = sp.csr_matrix(input_data)
                    print(f"Generated input {tensor_name} shape {shape}, nnz={dense_as_csr.nnz}")
                    dump_sparse_csr_fast(tensor_name, dense_as_csr, dump_dir)
                else:
                    print(f"Skipping input tensor {tensor_name} - unexpected shape {shape}")
                continue

            # Generate weight matrices (2D) with sparsity using fast scipy dump
            if len(shape) == 2:
                density = 1.0 - sparsity  # Convert sparsity to density
                sparse_mat = sp.random(shape[0], shape[1], density=density, format='csr', dtype=np.float32)
                sparse_mat.data *= 0.01  # Scale weights
                print(f"Generated weight {tensor_name} shape {shape}, density={density:.3f}, nnz={sparse_mat.nnz}")
                dump_sparse_csr_fast(tensor_name, sparse_mat, dump_dir)
            elif len(shape) == 1:
                # Bias vectors - generate random and dump fast
                bias_data = np.random.randn(shape[0]).astype(np.float32) * 0.01
                print(f"Generated bias {tensor_name} with shape {shape}")
                dump_dense_1d_fast(tensor_name, bias_data, dump_dir)


class autoencoder_nih_batched:
    """Data generator for NIH CXR batched autoencoder: batch=32, 65536 -> 256 -> 65536"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 5
        self.file_path = "/data/autoencoder_nih_batched"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.5

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"autoencoder_nih_batched collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        self.tensors[tensor_name] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format,
            'is_input': (input_arg == 0)
        }

    def generate_data(self, weight_sparsity=None):
        # Use weight_sparsity if provided, otherwise use self.sparsity
        sparsity = weight_sparsity if weight_sparsity is not None else self.sparsity
        print(f"Generating NIH batched autoencoder data with sparsity={sparsity}")
        dump_dir = os.getcwd() + self.file_path + "/"
        os.makedirs(dump_dir, exist_ok=True)

        for tensor_name, tensor_info in self.tensors.items():
            shape = tensor_info['shape']
            is_input = tensor_info.get('is_input', False)

            # Generate input tensor as dense (for standalone encoder/decoder ops)
            if is_input:
                if len(shape) == 2:
                    # Generate dense input batch (e.g., 32x1048576 for NIH)
                    input_data = np.random.randn(shape[0], shape[1]).astype(np.float32) * 0.1
                    dense_as_csr = sp.csr_matrix(input_data)
                    print(f"Generated input {tensor_name} shape {shape}, nnz={dense_as_csr.nnz}")
                    dump_sparse_csr_fast(tensor_name, dense_as_csr, dump_dir)
                else:
                    print(f"Skipping input tensor {tensor_name} - unexpected shape {shape}")
                continue

            # Generate weight matrices (2D) with sparsity using fast scipy dump
            if len(shape) == 2:
                density = 1.0 - sparsity  # Convert sparsity to density
                sparse_mat = sp.random(shape[0], shape[1], density=density, format='csr', dtype=np.float32)
                sparse_mat.data *= 0.01  # Scale weights
                print(f"Generated weight {tensor_name} shape {shape}, density={density:.3f}, nnz={sparse_mat.nnz}")
                dump_sparse_csr_fast(tensor_name, sparse_mat, dump_dir)
            elif len(shape) == 1:
                # Bias vectors - generate random and dump fast
                bias_data = np.random.randn(shape[0]).astype(np.float32) * 0.01
                print(f"Generated bias {tensor_name} with shape {shape}")
                dump_dense_1d_fast(tensor_name, bias_data, dump_dir)


class autoencoder_luna16_batched:
    """Data generator for Luna16 batched autoencoder: batch=32, 32768 -> 256 -> 32768"""
    def __init__(self, args):
        self.args = args
        self.tensors = {}
        self.input_args = 5
        self.file_path = "/data/autoencoder_luna16_batched"
        self.sparsity = args.sparsity if hasattr(args, 'sparsity') else 0.5

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        mode = [int(x) for x in mode]
        shape = [int(x) for x in shape]
        print(f"autoencoder_luna16_batched collecting: {tensor_name}, arg={input_arg}, shape={shape}")
        self.tensors[tensor_name] = {
            'name': tensor_name,
            'shape': shape,
            'mode': mode,
            'format': format,
            'is_input': (input_arg == 0)
        }

    def generate_data(self, weight_sparsity=None):
        # Use weight_sparsity if provided, otherwise use self.sparsity
        sparsity = weight_sparsity if weight_sparsity is not None else self.sparsity
        print(f"Generating Luna16 batched autoencoder data with sparsity={sparsity}")
        dump_dir = os.getcwd() + self.file_path + "/"
        os.makedirs(dump_dir, exist_ok=True)

        for tensor_name, tensor_info in self.tensors.items():
            shape = tensor_info['shape']
            is_input = tensor_info.get('is_input', False)

            # Generate input tensor as dense (for standalone encoder/decoder ops)
            if is_input:
                if len(shape) == 2:
                    # Generate dense input batch (e.g., 32x262144 for LUNA16)
                    input_data = np.random.randn(shape[0], shape[1]).astype(np.float32) * 0.1
                    dense_as_csr = sp.csr_matrix(input_data)
                    print(f"Generated input {tensor_name} shape {shape}, nnz={dense_as_csr.nnz}")
                    dump_sparse_csr_fast(tensor_name, dense_as_csr, dump_dir)
                else:
                    print(f"Skipping input tensor {tensor_name} - unexpected shape {shape}")
                continue

            # Generate weight matrices (2D) with sparsity using fast scipy dump
            if len(shape) == 2:
                density = 1.0 - sparsity  # Convert sparsity to density
                sparse_mat = sp.random(shape[0], shape[1], density=density, format='csr', dtype=np.float32)
                sparse_mat.data *= 0.01  # Scale weights
                print(f"Generated weight {tensor_name} shape {shape}, density={density:.3f}, nnz={sparse_mat.nnz}")
                dump_sparse_csr_fast(tensor_name, sparse_mat, dump_dir)
            elif len(shape) == 1:
                # Bias vectors - generate random and dump fast
                bias_data = np.random.randn(shape[0]).astype(np.float32) * 0.01
                print(f"Generated bias {tensor_name} with shape {shape}")
                dump_dense_1d_fast(tensor_name, bias_data, dump_dir)


# ============================================================================
# Partial and Unfused Autoencoder Data Generators
# These share the same data generation logic as the fully fused versions
# ============================================================================

class autoencoder_batched_encoder_fused(autoencoder_imagenet_batched):
    """Encoder fused: SpMM + Bias + ReLU for ImageNet dimensions"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_encoder_fused"

class autoencoder_batched_decoder_fused(autoencoder_imagenet_batched):
    """Decoder fused: SpMM + Bias for ImageNet dimensions"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_decoder_fused"

class autoencoder_batched_unfused_1_enc_spmm(autoencoder_imagenet_batched):
    """Unfused encoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_unfused_1_enc_spmm"

class autoencoder_batched_unfused_2_enc_bias(autoencoder_imagenet_batched):
    """Unfused encoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_unfused_2_enc_bias"

class autoencoder_batched_unfused_3_enc_relu(autoencoder_imagenet_batched):
    """Unfused encoder ReLU"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_unfused_3_enc_relu"

class autoencoder_batched_unfused_4_dec_spmm(autoencoder_imagenet_batched):
    """Unfused decoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_unfused_4_dec_spmm"

class autoencoder_batched_unfused_5_dec_bias(autoencoder_imagenet_batched):
    """Unfused decoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_batched_unfused_5_dec_bias"


# NIH partial and unfused models (inherit from autoencoder_nih_batched)
class autoencoder_nih_encoder_fused(autoencoder_nih_batched):
    """NIH Encoder fused: SpMM + Bias + ReLU"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_encoder_fused"

class autoencoder_nih_decoder_fused(autoencoder_nih_batched):
    """NIH Decoder fused: SpMM + Bias"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_decoder_fused"

class autoencoder_nih_unfused_1_enc_spmm(autoencoder_nih_batched):
    """NIH Unfused encoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_unfused_1_enc_spmm"

class autoencoder_nih_unfused_2_enc_bias(autoencoder_nih_batched):
    """NIH Unfused encoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_unfused_2_enc_bias"

class autoencoder_nih_unfused_3_enc_relu(autoencoder_nih_batched):
    """NIH Unfused encoder ReLU"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_unfused_3_enc_relu"

class autoencoder_nih_unfused_4_dec_spmm(autoencoder_nih_batched):
    """NIH Unfused decoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_unfused_4_dec_spmm"

class autoencoder_nih_unfused_5_dec_bias(autoencoder_nih_batched):
    """NIH Unfused decoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_nih_unfused_5_dec_bias"


# LUNA16 partial and unfused models (inherit from autoencoder_luna16_batched)
class autoencoder_luna16_encoder_fused(autoencoder_luna16_batched):
    """LUNA16 Encoder fused: SpMM + Bias + ReLU"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_encoder_fused"

class autoencoder_luna16_decoder_fused(autoencoder_luna16_batched):
    """LUNA16 Decoder fused: SpMM + Bias"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_decoder_fused"

class autoencoder_luna16_unfused_1_enc_spmm(autoencoder_luna16_batched):
    """LUNA16 Unfused encoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_unfused_1_enc_spmm"

class autoencoder_luna16_unfused_2_enc_bias(autoencoder_luna16_batched):
    """LUNA16 Unfused encoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_unfused_2_enc_bias"

class autoencoder_luna16_unfused_3_enc_relu(autoencoder_luna16_batched):
    """LUNA16 Unfused encoder ReLU"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_unfused_3_enc_relu"

class autoencoder_luna16_unfused_4_dec_spmm(autoencoder_luna16_batched):
    """LUNA16 Unfused decoder SpMM"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_unfused_4_dec_spmm"

class autoencoder_luna16_unfused_5_dec_bias(autoencoder_luna16_batched):
    """LUNA16 Unfused decoder bias add"""
    def __init__(self, args):
        super().__init__(args)
        self.file_path = "/data/autoencoder_luna16_unfused_5_dec_bias"


if __name__ == "__main__":
    dataset_name = 'Planetoid'  # The dataset class name (e.g., 'Planetoid', 'TUDataset', 'QM9#')
    root_dir = os.environ["DATA_PATH"]  # Root directory for the dataset
    name = 'Cora'  # Specific dataset name (e.g., 'Cora' for Planetoid)
