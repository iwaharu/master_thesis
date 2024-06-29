import networkx as nx
import scipy.sparse as sp
import numpy as np
import os
import tensorflow as tf

def importNet(fileName):
    ind = np.load(fileName, allow_pickle=True)

    dense_wMat = sp.csr_matrix((ind['wMat_data'],ind['wMat_indices'],ind['wMat_indptr']), shape=ind['wMat_shape']).toarray().astype(np.float32)
    dense_wMat[np.isnan(dense_wMat)]=0.

    aVec = ind['aVec'].astype(int)
    dense_aVec = np.zeros((len(aVec), 12)) # 0: dummy, 1~11: Linear ~ Squared
    for i in range(len(aVec)): dense_aVec[i, aVec[i]] = 1
    
    coo_wMat = sp.coo_array(dense_wMat)
    coo_aVec = sp.coo_array(dense_aVec)
    
    return coo_wMat, coo_aVec

def preprocess_graph(adj):
    """
    Disclaimer: originaly defined in the repo:
    https://github.com/deezer/gravity_graph_autoencoders
    """
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).todense().astype(np.float32)

    adj_label = adj_.todense().astype(np.float32)

    return adj_normalized, adj_label

def get_norm_pos_weight(adj):
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - np.sum(adj)) * 2)
    pos_weight = float(adj.shape[0] * adj.shape[0] - np.sum(adj)) / np.sum(adj)
    return norm, pos_weight

def load_data(dataset):
    assert type(dataset)!=list()

    files = []
    for d in dataset:
        if os.path.isfile(d) and d.endswith('_csr.npz'):
            files.append(d)
        elif os.path.isdir(d):
            files += [d+f for f in os.listdir(d) if f.endswith('_csr.npz')]

    adj, features = [], []
    max_num_nodes = 0
    for fname in files:
        iadj, ifeatures = importNet(fname)
        adj.append(iadj)
        features.append(ifeatures)
        max_num_nodes = max(max_num_nodes, iadj.shape[0])
    #adj = np.array([pad_adj(x, max_num_nodes) for x in adj])
    #features = np.array([pad_feature(x, max_num_nodes) for x in features])

    return adj, features, max_num_nodes

def serialize_sparse(densemx):
    spmx = tf.sparse.from_dense(densemx)

    indices_byte = tf.reshape(spmx.indices,[-1]).numpy().astype(np.int64).tobytes()
    shape_byte = spmx.dense_shape.numpy().astype(np.int64).tobytes()
    values_byte = spmx.values.numpy().astype(np.float32).tobytes()

    proto = tf.train.Example(features=tf.train.Features(feature={
        "indices": tf.train.Feature(bytes_list=tf.train.BytesList(value=[indices_byte])),
        "shape": tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape_byte])),
        "values": tf.train.Feature(bytes_list=tf.train.BytesList(value=[values_byte]))
    }))
    serialized = proto.SerializeToString()
    
    return serialized

def serialize_float(value):
    proto = tf.train.Example(features=tf.train.Features(feature={
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }))
    serialized = proto.SerializeToString()
    return serialized

def save_dataset(wannfile, save_to) -> None:
    from tqdm import tqdm

    adj, features, max_num_nodes = load_data(wannfile)

    os.makedirs(save_to, exist_ok=True)

    types = ['adj', 'features', 'adj_norm', 'adj_label', 'norm', 'pos_weight']
    writers = [tf.io.TFRecordWriter(save_to+'/'+x+'.tfrecord', tf.io.TFRecordOptions(compression_type="GZIP")) for x in types]
    shuffled_index = np.arange(len(adj))
    np.random.shuffle(shuffled_index)

    for i in tqdm(shuffled_index):
        adj_norm, adj_label = preprocess_graph(adj[i])
        data = [adj[i].todense(), features[i].todense(), adj_norm, adj_label]
        for j in range(len(data)):
            serialized = serialize_sparse(data[j])
            writers[j].write(serialized)

        norm, pos_weight = get_norm_pos_weight(adj[i])
        writers[-2].write(serialize_float(norm))
        writers[-1].write(serialize_float(pos_weight))

    for j in range(len(types)): writers[j].close()

    with open(save_to+'/max_node.out', mode='w') as f: f.write(str(max_num_nodes))

def save_conditional_dataset(filedict, save_to):
    from tqdm import tqdm
    # dict[taskname] = <list of wannfile>

    max_num_nodes = 0
    os.makedirs(save_to, exist_ok=True)
    types = ['adj', 'features', 'adj_norm', 'adj_label', 'norm', 'pos_weight', 'condition']
    writers = [tf.io.TFRecordWriter(save_to+'/'+x+'.tfrecord', tf.io.TFRecordOptions(compression_type="GZIP")) for x in types]

    tasks = list(filedict.keys())
    prob_labbeled = 0.6

    for t in range(len(tasks)):
        adj, features, num_nodes = load_data(filedict[tasks[t]])
        max_num_nodes = max(max_num_nodes, num_nodes)
        shuffled_index = np.arange(len(adj))
        np.random.shuffle(shuffled_index)

        for i in tqdm(shuffled_index):
            adj_norm, adj_label = preprocess_graph(adj[i])
            data = [adj[i].todense(), features[i].todense(), adj_norm, adj_label]
            for j in range(len(data)):
                serialized = serialize_sparse(data[j])
                writers[j].write(serialized)

            norm, pos_weight = get_norm_pos_weight(adj[i])
            writers[-3].write(serialize_float(norm))
            writers[-2].write(serialize_float(pos_weight))

            if np.random.rand()<prob_labbeled:
                writers[-1].write(serialize_float(float(t)))
            else:
                writers[-1].write(serialize_float(-1.0))

    for j in range(len(types)): writers[j].close()

    with open(save_to+'/max_node.out', mode='w') as f: f.write(str(max_num_nodes))
    with open(save_to+'/tasks.out', mode='w') as f: f.write(','.join(tasks))