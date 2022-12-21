import time
import json
import pickle
import torch
import random
import numpy as np
import pandas as pd
from os import makedirs
from os.path import exists
import torch.optim as optim
from Models import *


class KGE():
    """A class of multimodal Knowledge Graph Embedding."""
    
    def __init__(self, args):
        """
        (1) Initialize KGE with args dict.
        (2) Named model dir and out dir.
        (3) Load entity, relation and triplet.
        (4) Load model structure.
        """
        
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
        if self.model == 'ConvKB':
            self.eval_batch_size = 1
                                                                
        self.data_dir = 'KG/'
        self.out_dir = '{}{}/{} (C {}, D {})/'.format(self.data_dir,
                       self.model, self.margin, self.lambda_c, self.lambda_d)
        if not exists(self.out_dir):
            makedirs(self.out_dir)
                    
        print('\n' + '==' * 4 + ' < RDKG-115 & {} > '. \
              format(self.model) + '==' * 4) 
        self.load_data()        
        self.device = torch.device('cuda:' + self.gpu if
                                   torch.cuda.is_available() else 'cpu')
        self.MODEL = eval(self.model)(self.dim, self.lambda_c, self.lambda_d,
                                      self.margin, self.l2, self.n_E,
                                      self.n_R, self.device)
        self.MODEL.to(self.device)
        self.optimizer = optim.Adam(self.MODEL.parameters(), lr = self.l_r)
        
        print('\n    *Embedding Dim    : {}'.format(self.dim))
        print('    *Margin           : {}'.format(self.margin)) 
        print('    *Lambda C         : {}'.format(self.lambda_c))
        print('    *Lambda D         : {}\n'.format(self.lambda_d))
        for x, y in self.MODEL.named_parameters():
            print('    -{:18} : {}'.format(x, y.shape))
            
              
    def load_data(self):
        """
        (1) Get entity mapping dict (E_dict).
        (2) Get group mapping dict (G_dict).
        (3) Get relation mapping dict (R_dict).
        (4) Get train, dev and test dataset for embedding.
        (5) Get replace_h_prob dict and triple pool for negative 
            sample's generation.
        """
        
        self.E = list(pd.read_csv(self.data_dir + 'entity.csv')['E'])
        self.n_E = len(self.E)
        self.E_dict = dict(zip(self.E, range(self.n_E)))
        self.E_reverse_dict = dict(zip(range(self.n_E), self.E))
        
        self.G_dict = {}
        for e in self.E:
            g = e.split('-')[0]
            self.G_dict[g] = self.G_dict.get(g, []) + [self.E_dict[e]]
              
        self.R = list(pd.read_csv(self.data_dir + 'relation.csv')['R'])
        self.n_R = len(self.R)
        self.R_dict = dict(zip(self.R, range(self.n_R)))
        self.R_reverse_dict = dict(zip(range(self.n_R), self.R))
        
        p = self.data_dir + '_DATA.data'            
        if exists(p):
            DATA = mypickle(p)
        else:
            DATA = {}
            for key in ['train', 'dev', 'test']: 
                df = pd.read_csv(self.data_dir + key + '.csv')
                H, R, Ta = list(df['H']), list(df['R']), list(df['T'])
                t = [[self.E_dict[H[i]], self.R_dict[R[i]], self.E_dict[Ta[i]]]
                     for i in range(df.shape[0])]                            
                DATA[key] = np.array(t)
            mypickle(p, DATA)
                    
        for key in ['train', 'dev', 'test']:
            T = DATA[key]
            n_T = len(T)
            exec('self.{} = T'.format(key))
            exec('self.n_{} = n_T'.format(key))
            print('    #{:5} : {:7} ({:>5} E + {:2} R)'.format( \
                  key.title(), n_T, len(set(T[:, 0]) | set(T[:, 2])),
                  len(set(T[:, 1]))))
                            
        rpc_h_prob = {}
        for r in range(self.n_R):
            idx = np.where(self.train[:, 1] == r)[0]
            t_per_h = len(idx) / len(set(self.train[idx, 0]))
            h_per_t = len(idx) / len(set(self.train[idx, 2]))
            rpc_h_prob[r] = t_per_h / (t_per_h + h_per_t)
        self.rpc_h = lambda r : np.random.binomial(1, rpc_h_prob[r])
        
        self.pool = {tuple(x) for x in self.train.tolist() +
                     self.dev.tolist() + self.test.tolist()}
            

    def _train(self):  
        """
        (1) Training and Evalution process of embedding.
        (2) Calculate result of dev dataset, check whether reach the earlystop.
        """

        print('\n>>  Training Process.')
        print('    *Learning Rate    : {}'.format(self.l_r))
        print('    *l2 Rate          : {}'.format(self.l2))
        print('    *Train N Batch    : {}'.format(self.train_n_batch))
        print('    *Eval N Sample    : {}'.format(self.eval_n_sample))
        print('    *Eval Batch Size  : {}'.format(self.eval_batch_size))
        print('    *Max Epoches      : {}'.format(self.epoches))
        print('    *Earlystop Steps  : {}\n'.format(self.earlystop))

        if self.lambda_c or self.lambda_d:
            checkpoint = '{}{}/{} (C 0.0, D 0.0)/model.bin'. \
                        format(self.data_dir, self.model, self.margin)
            if not exists(checkpoint):
                raise('S Checkpoint not exists!')
            self.MODEL.load_checkpoint(checkpoint)
            
        eps = self.epoches
        bps = list(range(eps // 20 - 1, eps, eps // 20))
        print('    EPOCH  MRR  TOP1  TOP10 TOP100 KPI   time   Time  (Dev)')  
                    
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):
            for T_pos, T_neg in self.get_train_batches():
                self.optimizer.zero_grad()
                loss = self.MODEL.cal_loss(T_pos, T_neg)
                loss.backward()
                self.optimizer.step()
  
            if ep in bps:
                print('    {:^5} '.format(ep + 1), end = '')
                lp_out = self.link_prediction('dev')
                kpi = lp_out['KPI']
                _t = time.time()
                print(' {:^6.2f} {:^6.2f}'.format((_t - t1) / 60, 
                                                  (_t - t0) / 60), end = '')
                t1 = _t
            
                if ep == bps[0] or kpi > KPI[-1]:
                    print(' *')
                    if len(temp_kpi) > 0:
                        KPI.extend(temp_kpi)
                        temp_kpi = []
                    KPI.append(kpi)
                    torch.save(self.MODEL.state_dict(),
                               self.out_dir + 'model.bin',
                               _use_new_zipfile_serialization = False)
                    best_ep = bps[len(KPI) - 1] + 1                
                    result['dev-kpi'] = KPI
                    result['best-epoch'] = best_ep     
                    myjson(self.out_dir + 'result', result)
                else:
                    print('')
                    if len(temp_kpi) == self.earlystop:
                        break
                    else:
                        temp_kpi.append(kpi)
        
        if best_ep != eps:
            print('\n    Early stop at epoch of {} !'.format(best_ep))

    
    def get_train_batches(self):
        """
        (1) Generate batch data by train batch size.
        (2) Get negative triple (T_neg) for training.
        (3) Replace head or tail depends on replace_h_prob.
        """
                
        samples = list(range(self.n_train))
        np.random.shuffle(samples)
        
        n_batch = self.train_n_batch
        bs = self.n_train // n_batch
        idxes = [samples[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if self.n_train % bs != 0:
            idxes.append(samples[n_batch * bs: ])
        batches = []
        for idx in idxes:
            T_pos, T_neg = [], []
            for i in idx:
                T_pos.append(self.train[i])
                h, r, t = T_pos[-1]
                while True:
                    e = random.choice(range(self.n_E))
                    T = (e, r, t) if self.rpc_h(r) else (h, r, e)
                    if T not in self.pool:
                        T_neg.append(T)
                        break
            batches.append((torch.tensor(T_pos).long(),
                            torch.tensor(T_neg).long()))
        return batches
    

    def _predict(self):
        """Predict for test dataset."""
             
        print('\n>>  Predict Process.')
        self.MODEL.load_checkpoint(self.out_dir + 'model.bin')
        
        t0 = time.time()
        print('     MRR  TOP1  TOP10 TOP100 KPI  TIME  (Test)\n    ', end = '')
        lp_out = self.link_prediction('test')
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
        result = myjson(self.out_dir + 'result')        
        result.update(lp_out)
        myjson(self.out_dir + 'result', result)

        
    def link_prediction(self, key):   
        """
        Linking Prediction of knowledge graph embedding.
        Return entity MRR, TOP1, TOP10, TOP100
        """
        
        if key == 'dev':
            data = self.dev.copy()
            np.random.shuffle(data)
            data = data[: self.eval_n_sample]
        elif key == 'test':
            data = self.test
        
        bs = self.eval_batch_size
        ranks = []  
        with torch.no_grad():
            for k in [0, 2]:
                g_dic = {}
                for r in range(self.n_R):
                    tmp = data[data[:, 1] == r, :]
                    g = self.R_reverse_dict[r].split('->')[k // 2]
                    g_dic[g] = g_dic.get(g, []) + [tmp]
                for g, es in self.G_dict.items():
                    _data = np.vstack(g_dic[g])
                    left, right = es[0], es[-1]
                    idx = torch.tensor(range(left, right + 1)).to(self.device)
                    _data = torch.from_numpy(_data).long()
                    n_T = len(_data)        
                    n_batch = n_T // bs
                    Ts = [_data[i * bs: (i + 1) * bs] for i in range(n_batch)]
                    if n_T % bs != 0:
                        Ts.append(_data[n_batch * bs: ])
                    for T in Ts:            
                        scores = self.MODEL.cal_score(T, 'eval_h' if k == 0
                                 else 'eval_t', idx).cpu().data.numpy()
                        for i in range(len(T)):
                            sort = np.argsort(-scores[i]) + left
                            ranks.append(self.cal_rank(sort, T[i], k))
             
        if key == 'test':
            with open(self.out_dir + 'rank.txt', 'w') as file:
                for x in ranks:
                    file.writelines(str(x) + '\n')
                
        ranks = np.array(ranks)
        MRR = round(np.mean([1 / x for x in ranks]), 5)
        TOP1 = round(np.mean(ranks == 1), 5)
        TOP10 = round(np.mean(ranks <= 10), 5)
        TOP100 = round(np.mean(ranks <= 100), 5)
        KPI = round((MRR + TOP1 + TOP10 + TOP100) / 4, 5)
        
        print('{:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f}'. \
              format(MRR, TOP1, TOP10, TOP100, KPI), end = '')
        
        return {'MRR': MRR, 'TOP1': TOP1, 'TOP10': TOP10, 'TOP100': TOP100,
                'KPI': KPI}
        
    
    def cal_rank(self, sort, T, k):
        """
        Cal link prediction rank for a single triple,
        only entity match the relation type will be counted.
        """
        
        T = list(T.cpu().data.numpy())
        rank = 1 
        for x in sort:
            if x == T[k]:
                break
            else:
                new_T = T.copy()
                new_T[k] = x
                if tuple(new_T) not in self.pool:
                    rank += 1
        return rank

    
    def _evaluate(self):
        """Cal scores for 25 relations' all possible triplets."""

        print('\n>>  Evalution Process.')
        self.MODEL.load_checkpoint(self.out_dir + 'model.bin')
        _dir = self.out_dir + 'scores/'
        if not exists(_dir):
            makedirs(_dir)

        G = ['DI', 'DR', 'GP', 'PH', 'SM']
        with torch.no_grad():
            for X in G:
                idx_X = self.G_dict[X]
                n_X = len(idx_X) 
                for Y in G:
                    idx_Y = self.G_dict[Y]
                    n_Y = len(idx_Y)   
                    p = '{}{}->{}.data'.format(_dir, X, Y)
                    t0 = time.time()
                    if not exists(p):             
                        r = self.R_dict[X + '->' + Y]
                        M = np.zeros((n_X, n_Y), dtype = np.float32)
                        S = np.zeros((n_X, n_Y), dtype = np.float32)
                        if self.lambda_c:
                            C = np.zeros((n_X, n_Y), dtype = np.float32)
                        if self.lambda_d:
                            D = np.zeros((n_X, n_Y), dtype = np.float32)
                        for i in range(n_X):
                            h = idx_X[i]                        
                            T = torch.from_numpy(np.array([[h, r, t] for
                                                           t in idx_Y])).long()
                            score, score_s, score_c, score_d = \
                                self.MODEL.cal_score(T, 'eval')
                            M[i, :] = score.cpu().data.numpy()
                            S[i, :] = score_s.cpu().data.numpy()
                            if self.lambda_c:
                                C[i, :] = score_c.cpu().data.numpy()
                            if self.lambda_d:
                                D[i, :] = score_d.cpu().data.numpy()
                        result = {'M': M, 'S': S}
                        if self.lambda_c:
                            result['C'] = C
                        if self.lambda_d:
                            result['D'] = D
                        mypickle(p, result)
                    print('>>  {}->{} : {:5} * {:5} = {:9} ({:.2f} min)'. \
                          format(X, Y, n_X, n_Y, n_X * n_Y,
                                 (time.time() - t0) / 60))
    
    
    def run(self):
        """Runing Process"""
        
        for mode in ['train', 'predict', 'evaluate']:
            if eval('self.do_' + mode):
                eval('self._' + mode + '()')
    
    
    
def myjson(p, data = None):
    """Read (data is None) or Dump (data is not None) a json file."""

    if '.json' not in p:
        p += '.json'
    if data is None:
        with open(p) as file:
            return json.load(file)
    else:
        with open(p, 'w') as file:
            json.dump(data, file) 
            

def mypickle(p, data = None):
    """Read (data is None) or Dump (data is not None) a pickle file."""

    if '.data' not in p:
        p += '.data'
    if data is None:
        with open(p, 'rb') as file:
            return pickle.load(file)
    else:
        with open(p, 'wb') as file:
            pickle.dump(data, file)  