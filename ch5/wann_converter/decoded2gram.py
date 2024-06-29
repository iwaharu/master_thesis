import json, os, gc
import numpy as np
from tqdm import tqdm

class GramDecoder_1hot():
    def __init__(self, prefix):
        with open('gram/'+prefix+'/gram.json', mode='r') as f:
            gram = json.load(f)
        data = np.load('../gram/'+prefix+'/mask.npz')
        self.mask = data['mask'].astype(int)
        self.ind_of_ind = data['ind_of_ind']
        self.unique_lhs = gram['unique_lhs']
        self.grammar = gram['grams']

        self.lhs_map = {}
        for ix, lhs in enumerate(self.unique_lhs): self.lhs_map[lhs] = ix

    def _pop_or_nothing(self, a):
        try: return a.pop()
        except: return 'Nothing'
    
    def _is_nonterminal(self, a):
        return a in self.unique_lhs
    
    def _sample_using_masks(self, unmasked):
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self.unique_lhs[0])]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self.lhs_map[self._pop_or_nothing(a)] for a in S]
            mask = self.mask[next_nonterminal]
            masked_output = np.exp(unmasked[:,t,:])*mask + eps
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]),t,sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [filter(lambda a: self._is_nonterminal(str(a)), #str(a) not in ['output', 'None'],
                          self.grammar[i][1]) 
                   for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat

    def decode(self, unmasked):
        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [self.grammar[X_hat[0,t].argmax()] 
                     for t in range(X_hat.shape[1])] 
                    #for index in range(X_hat.shape[0])]
        return prod_seq

    def save(self, prefix, prod, label):
        with open(prefix+label+'.json', mode='w') as f:
            json.dump({'prods':prod}, f, indent=4)

    def convert_dir(self, dirname, savepath):
        for f in tqdm(os.listdir(dirname)):
            data = np.load(dirname+f, allow_pickle=True)
            #if os.path.exists(savepath+str(data['label'])+'.json'): continue
            #else: tqdm.write(str(data['label']))
            prod = self.decode(data['recon'])
            self.save(savepath, prod, str(data['label']))
            gc.collect()

class GramDecoder_w2v():
    def __init__(self, tag):
        with open('gram2vec/'+tag+'/metadata.tsv', 'r', encoding='utf-8') as m:
            metadata = [x.replace('\n','') for x in m.readlines()]
        vector = np.loadtxt('gram2vec/'+tag+'/vectors.tsv', delimiter='\t')
        self.table = {}
        for i in range(len(metadata)):
            if '->' not in metadata[i]:
                self.table['Nothing'] = vector[i,:]
                continue
            lhs, rhs = metadata[i].split('->')
            if self.table.get(lhs) is None:
                self.table[lhs] = dict()
            self.table[lhs][rhs] = vector[i,:]

    def _pop_or_nothing(self, a):
        try: return a.pop()
        except: return None
    
    def _nearest(self, lhs, target):
        cand = self.table[lhs]
        nearest_dest = np.inf
        nearest_rhs = None
        nearest_rhs = min(cand.keys(), key=lambda x:np.linalg.norm(cand[x] - target))
        return nearest_rhs.split(',')

    def decode(self, decoded):
        nSample, nGram = decoded.shape[0], decoded.shape[1]
        prod_seq = []

        for iSample in tqdm(range(nSample)):
            lhs = 's'
            que = []
            grams = []

            for iGram in tqdm(range(nGram), leave=False):
                rhs = self._nearest(lhs, decoded[iSample, iGram,:])
                #tqdm.write(lhs+'->'+str(rhs))
                grams.append((lhs, rhs))
                que += [x for x in rhs if not (x.startswith('output') or x=='deadend')]
                lhs = self._pop_or_nothing(que)
                if lhs is None: break

            prod_seq.append(grams)

        return prod_seq

    def save(self, prefix, prod):
        with open(prefix+'prod.json', mode='w') as f:
            json.dump({'prods':prod}, f, indent=4)
    
