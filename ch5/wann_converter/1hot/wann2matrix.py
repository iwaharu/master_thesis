from wann_src import *
from domain import *
import numpy as np
import json, os, pickle
from tqdm import tqdm
import networkx as nx
import tensorflow as tf

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

def matrix2graph(aVec, wKey, hyp):
    dim = int(aVec.shape[0])
    in_index = np.arange(games[hyp['task']].input_size+1)
    out_index = dim - games[hyp['task']].output_size + np.arange(games[hyp['task']].output_size)
    in_labels = ['bias'] + games[hyp['task']].in_out_labels[:games[hyp['task']].input_size]
    out_labels = games[hyp['task']].in_out_labels[games[hyp['task']].input_size:]

    graph = nx.DiGraph()
    for node in range(dim):
        if node in in_index:
            #val = 'input'
            idx = int(np.where(in_index==node)[0])
            val = 'input_' + str(in_labels[idx])
        elif node in out_index:
            #val = 'output'
            idx = int(np.where(out_index==node)[0])
            val = 'output_' + str(out_labels[idx])
        else:
            val = idx2act.get(aVec[node])
        graph.add_node(node, type=val, is_visited=False)

    graph.add_node(dim, type='S', is_visited=False)
    graph.add_edges_from([(dim, x) for x in in_index])
    graph.add_edges_from([(index//dim, index % dim) for index in wKey])

    return graph

def preorder_dfs(G, node):
    G.nodes[node]['is_visited'] = True
    result = [node]
    sorted_neighbor = sorted(G.neighbors(node), key=lambda x: actidx(G.nodes[x]['type']))
    for n in sorted_neighbor:
        #if G.nodes[n]['is_visited']==False:
        #duplicate two input nodes
        result += preorder_dfs(G, n)
    return result

def wann2gram(filename, hyp):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    _, aVec, wKey = importNet(filename, is_tqdm=True)
    graph = matrix2graph(aVec, wKey, hyp)
    #ordered_node = list(nx.dfs_preorder_nodes(graph, source=graph.number_of_nodes()-1))
    ordered_node = preorder_dfs(graph, graph.number_of_nodes()-1)
    ins = []
    gram = []
    for node in ordered_node:
        connected = [graph.nodes[x]['type'] for x in graph.neighbors(node)]
        #if graph.nodes[node]['type']=='S': continue
        if graph.nodes[node]['type']=='S' or graph.nodes[node]['type'].startswith('output'): continue

        elif graph.nodes[node]['type'].startswith('input'):
            if len(connected)==0: continue
            else:
                ins.append(graph.nodes[node]['type'])
                connected = tuple(sorted(connected, key = lambda x: actidx(x)))
                gram.append((graph.nodes[node]['type'], connected))

        else: #Hidden
            if len(connected)==0:
                gram.append((graph.nodes[node]['type'], ('Deadend',)))
            else:
                connected = tuple(sorted(connected, key = lambda x: actidx(x)))
                gram.append((graph.nodes[node]['type'], connected))
        
        #gram.append(('edge', tuple(['node']+['edge' for x in range(len(connected))])))
        #gram.append(('node', (graph.nodes[node]['type'], )))

    #gram.insert(0, ('S', tuple(['edge' for x in range(len(ins))]))) # S -> inputs
    gram.insert(0, ('S', tuple(ins)))

    return gram, len(gram)

### ==== make grammar ====

def make_mask(gram, unique_lhs, all_gram):
    mask = np.zeros((len(unique_lhs), all_gram), np.float32)
    for i in range(all_gram):
        index = unique_lhs.index(gram[i][0])
        mask[index,i] = 1
    ind_of_ind = np.where(mask!=0)[0]
    return mask.tolist(), ind_of_ind.tolist()

def save_gram_slave():
    global gramfile
    grams = set()
    MAX_USED_GRAM = 0

    while True:
        filename = comm.recv(source=0, tag=1)
        if filename is not None:
            hyp = comm.recv(source=0, tag=2)
            new_gram, used_gram = wann2gram(filename, hyp)
            new_gram = set(new_gram)
            grams = grams.union(new_gram)
            assert new_gram.issubset(grams)
            MAX_USED_GRAM = max(MAX_USED_GRAM, used_gram)

        else: # End signal recieved
            comm.send(grams, dest=0, tag=3)
            comm.send(MAX_USED_GRAM, dest=0, tag=4)
            print('Worker # ', rank, ' shutting down.')
            break

def save_gram_master(wanns):
    print('\t===\tPARSE GRAPH TO GRAMMAR\t===')
    global gramfile, nWorker
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
    
    print('gathering grams from workers')
    grams = set()
    MAX_USED_GRAM = 0
    for iWork in range(nSlave):
        comm.send(None, dest=(iWork)+1, tag=1)
        gram = comm.recv(source=(iWork)+1, tag=3)
        used_gram = comm.recv(source=(iWork)+1, tag=4)
        grams = grams.union(gram)
        MAX_USED_GRAM = max(MAX_USED_GRAM, used_gram)

    grams = [[f, list(t)] for f,t in grams]
    grams = sorted(grams, key = lambda x: actidx(x[0]))
    grams.append(['Nothing', ['None']]) # Dummy
    N_SAMPLE = len(wanns)
    ALL_GRAM = len(grams)
    unique_lhs = sorted(list(set([x[0] for x in grams])), key = lambda x: actidx(x))
    mask, ind_of_ind = make_mask(grams, unique_lhs, ALL_GRAM)
    with open(gramfile+'/gram.json', mode='w') as f: # +1 is for dummy
        json.dump({'sample_num':N_SAMPLE, 'all_gram':ALL_GRAM, 'max_used_gram':MAX_USED_GRAM+1, \
                    'unique_lhs':unique_lhs, 'grams':grams}, \
                    f, indent=4)
    np.savez_compressed(gramfile+'/mask', mask=mask, ind_of_ind=ind_of_ind)
    print('GRAM SAVED')

### ==== make vectors ====

def wann2gramvec(jsfile, filename, hyp):
    MAX_USED_GRAM = jsfile['max_used_gram']
    ALL_GRAM = jsfile['all_gram']
    gram = jsfile['grams']

    used_gram, _ = wann2gram(filename, hyp)
    used_gram = [[x, list(y)] for x,y in used_gram]

    dirs = filename.split('/')
    idx = int(dirs[2].replace('iter','')) * 10000 # padding
    if dirs[-1].endswith('_best.out'):
        idx += 10000
    else:
        idx += int(dirs[-1].split('_')[-1].replace('.out',''))
    label = dirs[1] + '_' + str(idx).zfill(5)

    onehot = np.zeros((MAX_USED_GRAM, ALL_GRAM), dtype=np.float32)
    for i in range(len(used_gram)):
        j = gram.index(used_gram[i])
        onehot[i,j] = 1

    onehot[len(used_gram), -1] = 1 # dummy
    arr_shape = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(onehot.shape, dtype=np.int64).tobytes()]))
    byte_arr = onehot.flatten().tobytes()
    arr_tensor = tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_arr]))


    label_tensor = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode("utf-8")]))
    proto = tf.train.Example(features=tf.train.Features(feature={
        "shape": arr_shape,
        "array": arr_tensor,
        "label": label_tensor
    }))
    serialized = proto.SerializeToString()

    return serialized

def wann2gramsparse(jsfile, filename, hyp):
    MAX_USED_GRAM = jsfile['max_used_gram']
    ALL_GRAM = jsfile['all_gram']
    gram = jsfile['grams']

    used_gram, _ = wann2gram(filename, hyp)
    used_gram = [[x, list(y)] for x,y in used_gram]

    dirs = filename.split('/')
    idx = int(dirs[2].replace('iter','')) * 10000 # padding
    if dirs[-1].endswith('_best.out'):
        idx += 10000
    else:
        idx += int(dirs[-1].split('_')[-1].replace('.out',''))
    label = dirs[1] + '_' + str(idx).zfill(5)

    row = list(range(MAX_USED_GRAM))
    col = [gram.index(used_gram[i]) for i in range(len(used_gram))] + [ALL_GRAM-1 for i in range(len(used_gram), MAX_USED_GRAM)]
    value = np.full((MAX_USED_GRAM,), 1., dtype=np.float32)
    
    proto = tf.train.Example(features=tf.train.Features(feature={
        "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=[MAX_USED_GRAM, ALL_GRAM])),
        "row": tf.train.Feature(int64_list=tf.train.Int64List(value=row)),
        "col": tf.train.Feature(int64_list=tf.train.Int64List(value=col)),
        #"channel": tf.train.Feature(int64_list=tf.train.Int64List(value=np.zeros(len(used_gram)+1, dtype=np.int64))),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=value)),
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode("utf-8")]))
    }))
    serialized = proto.SerializeToString()
    
    return serialized

def make_vec_slave():
    global gramfile
    with tf.io.TFRecordWriter(gramfile+'/data_sparse.tfrecords.'+str(rank-1), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        while True:
            jsfile = comm.recv(source=0, tag=1)
            if jsfile is not None:
                filename = comm.recv(source=0, tag=2)
                hyp = comm.recv(source=0, tag=3)

                #data = wann2gramvec(jsfile, filename, hyp)
                data = wann2gramsparse(jsfile, filename, hyp)
                writer.write(data)

            else: # End signal recieved
                print('Worker # ', rank, ' shutting down.')
                break

def make_vec_master(wanns):
    global gramfile, nWorker
    with open(gramfile+'/gram.json', mode='r') as f:
        jsfile = json.load(f)

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

                comm.send(jsfile, dest=(iWork)+1, tag=1)
                comm.send(filename, dest=(iWork)+1, tag=2)
                comm.send(hyp, dest=(iWork)+1, tag=3)
    
    print('stopping workers')
    for iWork in range(nSlave): comm.send(None, dest=(iWork)+1, tag=1)
    '''
    for i in tqdm(range(shard_num)):
        with tf.io.TFRecordWriter(gramfile+'_sparse.tfrecords.'+str(i), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
            for j in tqdm(shuffled_index[i], leave=False):
                #data = wann2gramvec(jsfile, wanns[j][1], wanns[j][0])
                data = wann2gramsparse(jsfile, wanns[j][1], wanns[j][0])
                writer.write(data)
    '''

def save_gram(wanns):
    # Launch main thread and workers
    if (rank == 0):
        save_gram_master(wanns)
    else:
        save_gram_slave()

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
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nWorker, rank)
    return "child"

def last_gen():
    dirs = ['log/multicolor/iter0/iter0_pop', 'log/mnist2352/iter1/iter1_pop',
            'log/grayscaleshape/iter2/iter2_pop', 'log/coloredmnist/iter0/iter0_pop',
            'log/multicolorshape/iter0/iter0_pop']#, 'log/mcsmnist/iter0/iter0_pop']
    wanns = []
    for d in dirs:
        pfile = d.split('/')[1]
        hyp = loadHyp(pFileName='p/default_wan.json')
        updateHyp(hyp, 'p/'+pfile+'.json')
        for f in tqdm(os.listdir(path=d)):
            wanns.append((hyp, d+'/'+f))
    
    global gramfile
    #gramfile = 'gram/last_gen/last_gen'
    gramfile = 'gram/task5_last'

    save_gram(wanns)
    make_vec(wanns)

def all_gen():
    dirs = [['log/multicolor/iter0/iter0_best'],
            ['log/mnist2352/iter0/iter0_best', 'log/mnist2352/iter1/iter1_best'],
            ['log/grayscaleshape/iter2/iter2_best'],
            ['log/coloredmnist/iter0/iter0_best'],
            ['log/multicolorshape/iter0/iter0_best'],
            ['log/mcsmnist/iter0/iter0_best']]
    wanns = []
    for ds in dirs:
        pfile = ds[0].split('/')[1]
        hyp = loadHyp(pFileName='p/default_wan.json')
        updateHyp(hyp, 'p/'+pfile+'.json')
        for d in ds:
            wanns += [(hyp, d+'/'+f) for f in os.listdir(path=d)]
            if 'grayscaleshape' not in d: wanns.append((hyp, d+'.out'))
    
    global gramfile
    gramfile = 'gram/all_gen'
    
    save_gram(wanns)
    make_vec(wanns)

def mc_mnist():
    dir = ['log/multicolor/iter0/iter0_pop', 'log/mnist2352/iter1/iter1_pop']
    wanns = []
    for d in dir:
        pfile = d.split('/')[1]
        hyp = loadHyp(pFileName='p/default_wan.json')
        updateHyp(hyp, 'p/'+pfile+'.json')
        wanns += [(hyp, d+'/'+f) for f in os.listdir(path=d)]

    global gramfile
    gramfile = 'gram/mc_mnist'

    save_gram(wanns)
    make_vec(wanns)

def mc():
    hyp = loadHyp(pFileName='p/default_wan.json')
    updateHyp(hyp, 'p/multicolor.json')
    d = 'log/multicolor/iter0/iter0_pop'
    wanns = [(hyp, d+'/'+f) for f in os.listdir(path=d)]

    global gramfile
    gramfile = 'gram/mc'

    save_gram(wanns)
    make_vec(wanns)