import numpy as np
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
(This file) Creating a dataset to feed to Word2Vec
-> Word2Vec skip-gram model
-> (This file) Creating embedding representations of wann
"""

import tensorflow as tf
import networkx as nx
import pandas as pd

from wann_src import *
from domain.config import games

import subprocess, math
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

idx2act = {1:'Linear', 2:'Unsigned_Step_Function', 3:'Sin', 4:'Gausian',
            5:'tanh', 6:'Sigmoid_unsigned', 7:'Inverse', 8:'Absolute_Value',
            9:'Relu', 10:'Cosine', 11:'Squared'}

def actidx(x):
    act2idx = {'S':-3, 'edge':-2, 'node':-1, 'input':0, 'Linear':1, 'Unsigned_Step_Function':2,
                'Sin':3, 'Gausian':4, 'tanh':5, 'Sigmoid_unsigned':6,
                'Inverse':7, 'Absolute_Value':8, 'Relu':9,
                'Cosine':10, 'Squared':11, 'output':12, 'Deadend':13, 'Nothing':14}
    idx = act2idx.get(x)
    if idx is None:
        if x.startswith('input'): return 0
        elif x.startswith('output'): return 12
    else:
        return idx

def generate(sequence):
    global vocab_size, sequence_length, window_size, num_ns
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                                    sequence,
                                    vocabulary_size=vocab_size,
                                    window_size=window_size,
                                    negative_samples=0)

    target_word, context_word = positive_skip_grams[0]
    context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                                            true_classes=context_class,  # class that should be sampled as 'positive'
                                            num_true=1,  # each positive skip-gram has 1 positive context class
                                            num_sampled=num_ns,  # number of negative context words to sample
                                            unique=True,  # all the negative samples should be unique
                                            range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
                                            name="negative_sampling"  # name of this operation
                                        )

    negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
    context = tf.concat([context_class, negative_sampling_candidates], 0)
    label = tf.constant([1] + [0]*num_ns, dtype="int64")
    
    target = tf.squeeze(target_word)
    context = tf.squeeze(context)
    label = tf.squeeze(label)

    return target, context, label

def matrix2graph(aVec, wKey, hyp):
    dim = int(aVec.shape[0])
    in_index = np.arange(games[hyp['task']].input_size+1)
    out_index = dim - games[hyp['task']].output_size + np.arange(games[hyp['task']].output_size)
    in_labels = ['bias'] + games[hyp['task']].in_out_labels[:games[hyp['task']].input_size]
    out_labels = games[hyp['task']].in_out_labels[games[hyp['task']].input_size:]

    graph = nx.DiGraph()
    for node in range(dim):
        if node in in_index:
            idx = int(np.where(in_index==node)[0])
            val = 'input_' + str(in_labels[idx])
        elif node in out_index:
            idx = int(np.where(out_index==node)[0])
            val = 'output_' + str(out_labels[idx])
        else:
            val = idx2act.get(aVec[node])
        graph.add_node(node, type=val)

    graph.add_node(dim, type='S')
    graph.add_edges_from([(dim, x) for x in in_index])
    graph.add_edges_from([(index//dim, index % dim) for index in wKey])

    return graph

def path2sen(graph, path):
    sen = []
    for n in path[:-1]:
        if graph.nodes[n]['type']=='S':
            connected = [graph.nodes[x]['type'] for x in graph.neighbors(n) if graph.out_degree(x)>0]
        else:
            connected = [graph.nodes[x]['type'] for x in graph.neighbors(n)]
        l = graph.nodes[n]['type'] + '->' + ','.join(sorted(connected, key = lambda x: actidx(x)))
        sen.append(l)
    if not graph.nodes[path[-1]]['type'].startswith('output'):
        sen.append(graph.nodes[path[-1]]['type'] + '->Deadend')
    return sen

def wann2sentence(filename, hyp):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    _, aVec, wKey = importNet(filename, is_tqdm=True)
    graph = matrix2graph(aVec, wKey, hyp)
    leaves = (v for v, d in graph.out_degree() if d==0 and not graph.nodes[v]['type'].startswith('input'))
    S = graph.number_of_nodes()-1
    all_path = list(nx.all_simple_paths(graph, S, leaves))

    sentences = []
    for path in all_path:
        sentences.append(' '.join(path2sen(graph, path)))
    return sentences

def save_vocab_sen_slave():
    global vecfile

    with tf.io.TFRecordWriter(vecfile+'/sentences.tfrecords.'+str(rank-1), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        while True:
            filename = comm.recv(source=0, tag=1)
            if filename is not None:
                hyp = comm.recv(source=0, tag=2)
                new_sentences = wann2sentence(filename, hyp)
                label = filename.split('/')[-1].replace('.out','')
            
                for sen in new_sentences:
                    proto = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode("utf-8")])),
                        "sentence": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sen.encode("utf-8")]))
                    }))
                    serialized = proto.SerializeToString()
                    writer.write(serialized)

            else: # End signal recieved
                print('Worker # ', rank, ' shutting down.')
                break

def load_sentences(dirname):
    description = {
        "label": tf.io.FixedLenFeature([], tf.string),
        "sentence": tf.io.FixedLenFeature([], tf.string)
    }

    def parse(example):
        parsed = tf.io.parse_single_example(example, description)
        sentence = parsed["sentence"]
        return sentence

    parent_dir = '/'.join(dirname.split('/')[:-1])
    tasks = dirname.split('/')[-1].split('_')
    files = []
    for task in tasks:
        files += [parent_dir+'/'+task+'/'+f for f in os.listdir(path=parent_dir+'/'+task) if 'sentences.tfrecords' in f]

    data = tf.data.TFRecordDataset(files, "GZIP").map(parse)
    return data

def save_vocab_sen_master(wanns):
    print('\t===\tPARSE GRAPH TO SENTENCES\t===')
    global nWorker, vocab_size
    nSlave = nWorker-1
    nJobs = len(wanns)
    nBatch = math.ceil(nJobs/nSlave) # First worker is master

    index = np.array_split(np.array(range(len(wanns))), nSlave)

    for iBatch in tqdm(range(nBatch), position=0): # Send one batch of individuals
        for iWork in range(nSlave): # (one to each worker if there)
            if iBatch < len(index[iWork]):
                i = index[iWork][iBatch]
                filename = wanns[i][1]
                hyp = wanns[i][0]

                comm.send(filename, dest=(iWork)+1, tag=1)
                comm.send(hyp, dest=(iWork)+1, tag=2)
    
    for iWork in range(nSlave): comm.send(None, dest=(iWork)+1, tag=1)

    dataset = load_sentences(vecfile)
    vectorize_layer = tf.keras.layers.TextVectorization(
                            standardize='lower',
                            split='whitespace',
                            max_tokens=vocab_size,
                            output_mode='int')
    vectorize_layer.adapt(dataset.batch(1024))

    with open(vecfile+'/vocab_'+str(vocab_size)+'.obj', 'wb') as f:
        pickle.dump(vectorize_layer.get_vocabulary(), f)

def save_w2v_dataset_slave():
    global vecfile

    with tf.io.TFRecordWriter(vecfile+'/w2v_dataset.tfrecords.'+str(rank-1), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        while True:
            sequence = comm.recv(source=0, tag=1)
            if sequence is not None:
                target, context, label = generate(sequence)

                #((target, context), label)
                proto = tf.train.Example(features=tf.train.Features(feature={
                    "target": tf.train.Feature(int64_list=tf.train.Int64List(value=[target])),
                    "context": tf.train.Feature(int64_list=tf.train.Int64List(value=context)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                }))
                serialized = proto.SerializeToString()
                writer.write(serialized)

            else: # End signal recieved
                print('Worker # ', rank, ' shutting down.')
                break

def save_w2v_dataset_master():
    print('\t===\tPARSE GRAPH TO VOCABLUARY\t===')
    global vecfile, nWorker, vocab_size, sequence_length

    vectorize_layer = tf.keras.layers.TextVectorization(
                            standardize='lower',
                            split='whitespace',
                            max_tokens=vocab_size,
                            output_mode='int')
                
    with open(vecfile+'/vocab_'+str(vocab_size)+'.obj', 'rb') as f:
        vocab = pickle.load(f)
        vectorize_layer.set_vocabulary(vocab)

    dataset = load_sentences(vecfile)
    text_vector_ds = dataset.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()

    nSlave = nWorker-1 # First worker is master
    iWork = 0
    for sequence in tqdm(text_vector_ds.take(-1)):
        comm.send(sequence, dest=(iWork)+1, tag=1)
        iWork += 1
        if iWork==nSlave: iWork=0
    
    for iWork in range(nSlave):
        comm.send(None, dest=(iWork)+1, tag=1)

def wann2array(vector, filename, hyp):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    _, aVec, wKey = importNet(filename, is_tqdm=True)
    graph = matrix2graph(aVec, wKey, hyp)
    leaves = (v for v, d in graph.out_degree() if d==0 and not graph.nodes[v]['type'].startswith('input'))
    S = graph.number_of_nodes()-1
    all_path = list(nx.all_simple_paths(graph, S, leaves))

    arr = []
    for path in all_path:
        arr += [vector[s.lower()] for s in path2sen(graph, path)]
    arr = np.array(arr, dtype=np.float32)
    #print(arr.shape)

    dirs = filename.split('/')
    idx = dirs[-1].split('_')[-1].replace('.out','')
    label = dirs[1] + '_' + idx.zfill(5)

    arr_shape = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(arr.shape, dtype=np.int64).tobytes()]))
    byte_arr = arr.flatten().tobytes()
    arr_tensor = tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_arr]))


    label_tensor = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode("utf-8")]))
    proto = tf.train.Example(features=tf.train.Features(feature={
        "shape": arr_shape,
        "array": arr_tensor,
        "label": label_tensor
    }))
    serialized = proto.SerializeToString()

    return serialized, arr.shape[0]
def make_vec_slave():
    global vecfile
    MAX_USED_GRAM = 0
    with tf.io.TFRecordWriter(vecfile+'/mnist_data.tfrecords.'+str(rank-1), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        while True:
            table = comm.recv(source=0, tag=1)
            if table is not None:
                filename = comm.recv(source=0, tag=2)
                hyp = comm.recv(source=0, tag=3)

                data, used_gram = wann2array(table, filename, hyp)
                MAX_USED_GRAM = max(MAX_USED_GRAM, used_gram)
                writer.write(data)

            else: # End signal recieved
                comm.send(MAX_USED_GRAM, dest=0, tag=4)
                print('Worker # ', rank, ' shutting down.')
                break

def make_vec_master(wanns):
    global vecfile, nWorker

    with open(vecfile+'/metadata.tsv', 'r', encoding='utf-8') as m:
        metadata = [x.replace('\n','') for x in m.readlines()]
    vector = np.loadtxt(vecfile+'/vectors.tsv', delimiter='\t')
    table = {}
    for i in range(len(metadata)): table[metadata[i]] = vector[i,:]

    print('===\tCONVERT GRAPH TO GRAM ARRAY\t===')
    nSlave = nWorker-1
    nJobs = len(wanns)
    nBatch = math.ceil(nJobs/nSlave) # First worker is master

    shuffled_index = np.array(range(len(wanns)))
    #np.random.shuffle(shuffled_index)
    shuffled_index = np.array_split(shuffled_index, nSlave)

    for iBatch in tqdm(range(nBatch), position=1): # Send one batch of individuals
        for iWork in range(nSlave): # (one to each worker if there)
            if iBatch < len(shuffled_index[iWork]):
                i = shuffled_index[iWork][iBatch]
                filename = wanns[i][1]
                hyp = wanns[i][0]

                comm.send(table, dest=(iWork)+1, tag=1)
                comm.send(filename, dest=(iWork)+1, tag=2)
                comm.send(hyp, dest=(iWork)+1, tag=3)

    print('stopping workers')
    MAX_USED_GRAM = 0
    for iWork in range(nSlave):
        comm.send(None, dest=(iWork)+1, tag=1)
        used_gram = comm.recv(source=(iWork)+1, tag=4)
        MAX_USED_GRAM = max(MAX_USED_GRAM, used_gram)

    print('MAX_USED_GRAM:', MAX_USED_GRAM)


def save_vocab_sen(wanns):
    # Launch main thread and workers
    if (rank == 0):
        save_vocab_sen_master(wanns)
    else:
        save_vocab_sen_slave()

def save_w2v_dataset():
    # Launch main thread and workers
    if (rank == 0):
        save_w2v_dataset_master()
    else:
        save_w2v_dataset_slave()

def make_vec(wanns):
    # Launch main thread and workers
    if (rank == 0):
        make_vec_master(wanns)
    else:
        make_vec_slave()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    #subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    subprocess.check_call(["mpiexec", "-n", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nWorker, rank)
    return "child"

def mnist():
    hyp = loadHyp(pFileName='p/default_wan.json')
    updateHyp(hyp, 'p/mnist2352.json')
    d = 'log/mnist2352/iter1/iter1_pop'
    wanns = [(hyp, d+'/'+f) for f in os.listdir(path=d)]

    global vecfile, vocab_size
    vecfile = 'gram2vec/mnist'
    vocab_size = 5200

    save_vocab_sen(wanns)

def mc_mnist():
    dir = ['log/mnist2352/iter1/iter1_pop'] #'log/multicolor/iter0/iter0_pop'
    wanns = []
    for d in dir:
        pfile = d.split('/')[1]
        hyp = loadHyp(pFileName='p/default_wan.json')
        updateHyp(hyp, 'p/'+pfile+'.json')
        wanns += [(hyp, d+'/'+f) for f in os.listdir(path=d)]

    global vecfile, vocab_size, sequence_length, window_size, num_ns
    vecfile = 'gram2vec/mc_mnist'
    vocab_size, sequence_length = 7000, 40
    window_size, num_ns = 3, 4

    #save_w2v_dataset()
    make_vec(wanns)

def mc():
    hyp = loadHyp(pFileName='p/default_wan.json')
    updateHyp(hyp, 'p/multicolor.json')
    d = 'log/multicolor/iter0/iter0_pop'
    wanns = [(hyp, d+'/'+f) for f in os.listdir(path=d)]

    global vecfile, vocab_size, sequence_length, window_size, num_ns
    vecfile = 'gram2vec/mc'
    vocab_size, sequence_length = 2500, 10
    window_size, num_ns = 2, 4

    #save_vocab_sen(wanns)
    save_w2v_dataset()
    #make_vec(wanns)

if __name__ == "__main__":
    if "parent" == mpi_fork(26): os._exit(0)
    #mc()
    #mnist()
    mc_mnist()