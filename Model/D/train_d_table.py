import sys
import time
import torch
import random
import numpy as np
import pandas as pd
import tokenization as tkz
from os import makedirs
from os.path import exists
from DTModel import DTModel
from optimization import AdamW, warmup_schedule
sys.path.append('../')
from train_kge import myjson, mypickle


class D_Table():
    """Train D table by BERT structure."""

    def __init__(self, args):
        """
        (1) Initialized D_table as args dict.
        (2) Load train and dev dataset.
        (3) Construct BERT model.
        """

        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))

        self.model_dir = '../Pretrained BERT/' + self.model + '/'
        self.out_dir = self.model + '/'
        if not exists(self.out_dir):
            makedirs(self.out_dir)
            
        config = myjson(self.model_dir + 'config')

        print('\n' + '==' * 4 + ' < {} for d table > '. \
              format(self.model) + '==' * 4) 
        self.load_data()

        print('\n    #length of sequence : {}'.format(self.len_d))        
        print('    *Learning_Rate      : {}'.format(self.l_r))
        print('    *Batch_Size         : {}'.format(self.batch_size))
        print('    *Max_Epoch          : {}'.format(self.epoches))
        print('    *Earlystop Steps    : {}'.format(self.earlystop))
        self.bert_dt = DTModel(config, self.n_R)
        self.device = torch.device('cuda:' + self.gpu 
                                    if torch.cuda.is_available() else 'cpu')
        self.bert_dt.to(self.device)

        param_op = list(self.bert_dt.named_parameters())
        no_decay = ['bias', 'layerNorm']
        grouped_paras = \
            [{'params': [p for n, p in param_op if not 
                any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
             {'params': [p for n, p in param_op if 
                any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]
        self.optimizer = AdamW(grouped_paras, lr = self.l_r)
        n_step = (self.n_train // self.batch_size + 1) * self.epoches
        self.scheduler = warmup_schedule(self.optimizer, n_step // 10, n_step)


    def load_data(self):
        """
        (1) Load _all triple and delete relation, then drop duplicates.
        (2) Generate entity index dict.
        (3) Generate description dict.
        (4) Generate train (450,000) and dev (50,000) dataset.
        """
        
        E = list(pd.read_csv('../KG/entity.csv')['E'])
        self.n_E = len(E)
        E_dict = dict(zip(E, range(self.n_E)))
        print('    #entity   : {}'.format(self.n_E))
        R = list(pd.read_csv('../KG/relation.csv')['R'])
        self.n_R = len(R)
        R_dict = dict(zip(R, range(self.n_R)))
        print('    #relation : {}'.format(self.n_R))
        
        train_df = pd.read_csv('../KG/train.csv')
        H = [E_dict[h] for h in list(train_df['H'])]
        R = [R_dict[r] for r in list(train_df['R'])] 
        T = [E_dict[t] for t in list(train_df['T'])]
        n_train_df = train_df.shape[0]
        print('\n    #train_csv : {}'.format(n_train_df))
        
        self.n_train, self.n_dev = 450000, 50000
        p1 = 'train_dev_idx.data'            
        if not exists(p1):
            e_dict = {}
            for i in random.sample(range(n_train_df), n_train_df):
                for e in [H[i], T[i]]: 
                    e_dict[e] = e_dict.get(e, []) + [i]
            idxes, pool = [[[], []], []]
            for ys in list(e_dict.values()):
                k = 0
                for y in ys:
                    if y not in pool:
                        pool.append(y)
                        idxes[k].append(y)
                        k += 1
                        if k == 2:
                            break
            train_idx, dev_idx = idxes
            rest_idx = list(set(range(n_train_df)) - set(pool))
            while len(train_idx) < self.n_train:
                train_idx.append(rest_idx.pop())
            while len(dev_idx) < self.n_dev:
                dev_idx.append(rest_idx.pop())
            mypickle(p1, {'train': train_idx, 'dev': dev_idx})
        idxes = mypickle(p1)
        
        p2 = self.out_dir + 'D_dict.data' 
        if not exists(p2):
            self.tokenizer = tkz.Tokenizer(self.model_dir + 'vocab.txt',
                                           self.model != 'biobert')
            self.D_dict = []
            A = myjson('../../Annotation/E_dict')
            for e in E:
                _token = self.tokenizer.tokenize( \
                            tkz.convert_to_unicode(A[e]['D']))
                if len(_token) > self.len_d - 2:
                    _token = _token[: (self.len_d - 2)]
                _token = ['[CLS]'] + _token + ['[SEP]']
                token = self.tokenizer.convert_tokens_to_ids(_token)
                while len(token) < self.len_d:
                    token.append(0)
                self.D_dict.append(token)   
            mypickle(p2, self.D_dict)    
        self.D_dict = mypickle(p2)
                       
        for key in ['train', 'dev']:
            idx = idxes[key]
            _ = np.array([[self.D_dict[H[i]], self.D_dict[T[i]]] for i in idx])
            exec('self.{}_E = _'.format(key))
            _ = np.array([R[i] for i in idx])
            exec('self.{}_R = _'.format(key))
            print('    #{:5}    : {}'.format(key, len(idx)))


    def _train(self):
        """
        (1) Training process of D_table's generating.
        (2) Evaluate for dev dataset each epoch.
        """
        
        print('\n>>  Training Process.')
        self.bert_dt.load_checkpoint(self.model_dir + 'pytorch_model.bin',
                                    'pretrain')
        
        dev_batches = self.get_batches('dev')
        print('    EPOCH TRAIN-LOSS DEV-LOSS  time   TIME (min)')
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(self.epoches):
            print('    {:^5}'.format(ep + 1), end = '')
            train_batches = self.get_batches('train')
            train_loss = []
            for token, r in train_batches:
                self.optimizer.zero_grad()
                token = token.to(self.device)
                r = r.to(self.device)
                loss = self.bert_dt(token, self.batch_size, r, mode = 'train')
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss.append(loss.item())
            train_loss = round(np.sum(train_loss) / self.n_train, 4)    

            dev_loss = []
            with torch.no_grad():
                for token, r in dev_batches:
                    token = token.to(self.device)
                    r = r.to(self.device)
                    loss = self.bert_dt(token, self.batch_size, r,
                                        mode = 'eval')
                    dev_loss.append(loss.item())
            kpi = round(np.sum(dev_loss) / self.n_dev, 4)    

            _t = time.time()
            print(' {:^10.4f} {:^8.4f} {:^6.2f} {:^6.2f}'.format(train_loss,
                  kpi, (_t - t1) / 60, (_t - t0) / 60), end = '')
            t1 = _t
            
            if ep == 0 or kpi < KPI[-1]:
                print(' *')
                if len(temp_kpi) > 0:
                    KPI.extend(temp_kpi)
                    temp_kpi = []
                KPI.append(kpi)
                torch.save(self.bert_dt.state_dict(),
                           self.out_dir + 'model.bin',
                           _use_new_zipfile_serialization = False)
                result['dev-kpi'] = KPI
                result['best-epoch'] = len(KPI)
                myjson(self.out_dir + 'result', result)
            else:
                print('')
                if len(temp_kpi) == self.earlystop:
                    break
                else:
                    temp_kpi.append(kpi)
        
        result['KPI'] = KPI[-1]
        myjson(self.out_dir + 'result', result)
                
        if ep != self.epoches - 1:
            print('\n    Early stop at epoch of {} !'.format(len(KPI)))            
    

    def _predict(self):
        """Prediction process of D_table's generating."""
               
        print('\n>>  Predict Process.')
        self.bert_dt.load_checkpoint(self.out_dir + 'model.bin', 'predict')
        
        bs = self.batch_size
        n_batch = self.n_E // bs
        tokens = [self.D_dict[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if n_batch * bs != self.n_E:
            tokens.append(self.D_dict[n_batch * bs: ])

        t0 = time.time()
        D_table = []
        with torch.no_grad():
            for token in tokens:
                token = torch.tensor(token).long().to(self.device)
                D_table.append(self.bert_dt(token, None, None, 'predict'). \
                               cpu().data.numpy())
            
        mypickle(self.out_dir + 'D_table', np.vstack(D_table))
        print('    Total Time: {:.2f}min'.format((time.time() - t0) / 60))
        
        
    def get_batches(self, key):
        """
        Get batch example for train and dev examples with args' batch_size.
        """

        bs = self.batch_size
        n = eval('self.n_' + key)
        sample = random.sample(range(n), n)
        idxes = [sample[i * bs: (i + 1) * bs] for i in range(n // bs)]
        E = eval('self.' + key + '_E')
        R = eval('self.' + key + '_R')
        batches = []
        for idx in idxes:
            batches.append( \
                    (torch.tensor(np.vstack([E[idx, 0], E[idx, 1]])).long(),
                     torch.tensor(R[idx]).long()))
        return batches