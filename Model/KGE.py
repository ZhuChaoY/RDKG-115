import os
import re
import json
import time
import pickle
import random
import collections
import numpy as np
import pandas as pd
import tensorflow as tf


class KGE():
    """
    A class of processing and tool functions for Knowledge Graph Embedding.
    Abbreviation:
        c(C) : Category(s) 
        d(D) : Description(s)  
        s(S) : Structure(s)
        t(T) : Triple(s)
        r(R) : Relation(s)
        e(E) : Entity(ies)
        h(H) : Head(s)
        t(T)a: Tail(s)
        p    : path
    """
    
    def __init__(self, args):
        """
        (1) Initialize KGE with args dict.
        (2) Named model dir and out dir.
        (3) Load entity, relation and triple.
        (4) Load common model structure.
        """
        
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                        
        self.data_dir = '../Model/KG/'
        self.out_dir = '{}{}/{} (C {}, D {})/'.format(self.data_dir,
                       self.model, self.margin, self.lambda_c, self.lambda_d)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
                    
        print('\n\n' + '==' * 4 + ' < {} > '.format(self.model) + '==' * 4)        
        self.em_data()
        self.common_structure()
    
              
    def em_data(self):
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
        if os.path.exists(p):
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
                            
        rpc_h_prob = {} #Bernoulli Trick
        for r in range(self.n_R):
            idx = np.where(self.train[:, 1] == r)[0]
            t_per_h = len(idx) / len(set(self.train[idx, 0]))
            h_per_t = len(idx) / len(set(self.train[idx, 2]))
            rpc_h_prob[r] = t_per_h / (t_per_h + h_per_t)
        self.rpc_h = lambda r : np.random.binomial(1, rpc_h_prob[r])
        
        self.pool = {tuple(x) for x in self.train.tolist() +
                     self.dev.tolist() + self.test.tolist()}
    
    
    def common_structure(self):
        """The common structure of KGE model."""
        
        print('\n    *Embedding Dim    : {}'.format(self.dim))
        print('    *Margin           : {}'.format(self.margin)) 
        print('    *Lambda_C         : {}'.format(self.lambda_c))
        print('    *Lambda_D         : {}'.format(self.lambda_d))
        print('    *l2 Rate          : {}'.format(self.l2))
        print('    *Learning_Rate    : {}'.format(self.l_r))
        print('    *Train_Batch_Size : {}'.format(self.train_batch_size))
        print('    *Eval_Batch_Size  : {}'.format(self.eval_batch_size))
        print('    *Max Epoches      : {}'.format(self.epoches))
        print('    *Earlystop Steps  : {}'.format(self.earlystop))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 2])
        self.K = np.sqrt(6.0 / self.dim)
        
        with tf.variable_scope('structure'): 
            self.E_table = tf.get_variable('entity_table', initializer = \
                  tf.random_uniform([self.n_E, self.dim], -self.K, self.K))
            self.E_table = tf.nn.l2_normalize(self.E_table, 1)    
            if self.model == 'RotatE':
                pi = 3.1415926
                R_table = tf.get_variable('relation_table', initializer = \
                          tf.random_uniform([self.n_R, self.dim], -pi, pi))
            else:
                R_table = tf.get_variable('relation_table', initializer = \
                      tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
                R_table = tf.nn.l2_normalize(R_table, 1)
            
            h_pos = tf.gather(self.E_table, self.T_pos[:, 0])
            t_pos = tf.gather(self.E_table, self.T_pos[:, -1])
            h_neg = tf.gather(self.E_table, self.T_neg[:, 0])
            t_neg = tf.gather(self.E_table, self.T_neg[:, -1])
            r = tf.gather(R_table, self.T_pos[:, 1])

            self.l2_kge = [h_pos, t_pos, r, h_neg, t_neg]
            self.kge_variables()
            s_pos = self.em_structure(h_pos, r, t_pos, 'pos')
            self.score_pos = self.cal_score(s_pos)
            s_neg = self.em_structure(h_neg, r, t_neg, 'neg')
            score_neg = self.cal_score(s_neg)

        if self.lambda_c:
            with tf.variable_scope('category'):
                self.C_table = self.C_layer()
                self.h_c_pos = tf.gather(self.C_table, self.T_pos[:, 0])
                self.t_c_pos = tf.gather(self.C_table, self.T_pos[:, -1])
                h_c_neg = tf.gather(self.C_table, self.T_neg[:, 0])
                t_c_neg = tf.gather(self.C_table, self.T_neg[:, -1])        
                c_pos = self.h_c_pos - self.t_c_pos
                c_neg = h_c_neg - t_c_neg  
                s_c_pos = self.projector(s_pos, c_pos)
                s_c_neg = self.projector(s_neg, c_neg)
                self.score_pos -= self.lambda_c * self.cal_score(s_c_pos)
                score_neg -= self.lambda_c * self.cal_score(s_c_neg)
                
        if self.lambda_d:
            with tf.variable_scope('description'):
                self.D_table = self.D_layer()
                self.h_d_pos = tf.gather(self.D_table, self.T_pos[:, 0])
                self.t_d_pos = tf.gather(self.D_table, self.T_pos[:, -1])
                h_d_neg = tf.gather(self.D_table, self.T_neg[:, 0])
                t_d_neg = tf.gather(self.D_table, self.T_neg[:, -1])
                d_pos = self.h_d_pos - self.t_d_pos
                d_neg = h_d_neg - t_d_neg                       
                s_d_pos = self.projector(s_pos, d_pos)
                s_d_neg = self.projector(s_neg, d_neg)
                self.score_pos -= self.lambda_d * self.cal_score(s_d_pos)
                score_neg -= self.lambda_d * self.cal_score(s_d_neg)
                
        with tf.variable_scope('loss'): 
            loss = tf.reduce_sum(tf.nn.relu(self.margin + \
                   self.score_pos - score_neg))
            loss_kge = tf.add_n([tf.nn.l2_loss(v) for v in self.l2_kge])
            loss = loss + self.l2 * loss_kge
            self.train_op = tf.train.AdamOptimizer(self.l_r).minimize(loss)
                            
        with tf.variable_scope('link_prediction'): 
            self.lp_h, self.lp_t = self.cal_lp_score(h_pos, r, t_pos)


    def C_layer(self):
        """Category Embedding layer"""
        
        all_C = mypickle('C&D/C_dict')
        E_index = myjson('C&D/E_index')
        C_dict = [all_C[E_index[e]] for e in self.E]
        c_list = []
        for c in C_dict:
            c_list.extend(c)
        
        c_list = sorted(set(c_list))
        n_C = len(c_list)
        c_dict = dict(zip(c_list, range(n_C))) 
        
        C_dict = tf.to_int32([[c_dict[x] for x in y] for y in C_dict])
        dim = self.dim if self.model != 'RotatE' else self.dim * 2
        raw_C_table = tf.get_variable('category_table', initializer = \
                      tf.random_uniform([n_C, dim], -self.K, self.K))
        raw_C_table = tf.nn.l2_normalize(raw_C_table, 1)
        C_table = tf.reduce_mean(tf.gather(raw_C_table, C_dict), 1)
        
        return C_table
        

    def D_layer(self):
        """Description Embedding layer"""
        
        all_D = mypickle('C&D/D_table')
        E_index = myjson('C&D/E_index')
    
        init_D = np.array([all_D[E_index[e]] for e in self.E])        
        raw_D_table = tf.get_variable('description_table', [self.n_E, 768],
                      initializer = tf.constant_initializer(init_D))
        raw_D_table = tf.nn.l2_normalize(raw_D_table, 1)
        dim = self.dim if self.model != 'RotatE' else 2 * self.dim
        K = np.sqrt(6.0 / (768 + dim))
        W = tf.get_variable('d_weight', initializer = \
                            tf.random_uniform([768, dim], -K, K))
        D_table = tf.matmul(raw_D_table, W)
        
        return D_table


    def projector(self, s, p):
        """
        Reverse hyperplane projection for C and D layer.
        
        Args:
            s : score normal vector
            p : project vector
        """
        
        s = tf.nn.l2_normalize(s, 1)
        return p - tf.reduce_sum(s * p, -1, keepdims = True) * s


    def em_train(self, sess):  
        """
        (1) Initialize and display variables and shapes.
        (2) Training and Evalution process of embedding.
        (3) Calculate result of dev dataset, check whether reach the earlystop.
            
        Args:
            sess: tf.Session
        """

        eps = self.epoches
        bps = list(range(eps // 20 - 1, eps, eps // 20))
        print('    EPOCH   MR    MRR  TOP1  TOP5  TOP10 TOP100 ' \
              'time   Time  (Dev)')  
            
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):  
            T_poss = self.get_batches('train')
            for T_pos in T_poss:
                T_neg = self.get_T_neg(T_pos)
                _ = sess.run(self.train_op, {self.T_pos: T_pos,
                                             self.T_neg: T_neg})     
            
            if ep in bps:
                print('    {:^5} '.format(ep + 1), end = '')
                lp_out = self.link_prediction(sess, 'dev')
                kpi = lp_out['TOP100']
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
                    tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
                    best_ep = bps[len(KPI) - 1] + 1                
                    result['dev-top100'] = KPI
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

    
    def get_batches(self, key):
        """
        Generate batch data by batch size.
        
        args:
            key: 'train', 'dev' or 'test'
        """
        
        bs = self.train_batch_size if key == 'train' else self.eval_batch_size
        data = eval('self.' + key + '.copy()')
        if key != 'test':
            np.random.shuffle(data)
        if key == 'dev':
            data = data[: len(data) // 2]
                
        n = len(data)
        n_batch = n // bs
        T_poss = [data[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if n % bs != 0:
            T_poss.append(data[n_batch * bs: ])
        return T_poss 
    

    def get_T_neg(self, T_pos):
        """
        (1) Get negative triple (T_neg) for training.
        (2) Replace head or tail depends on replace_h_prob.
        
        Args:
            T_pos: positive triples
        """
        
        T_negs = []
        for h, r, ta in T_pos.tolist():
            while True:    
                new_e = random.choice(range(self.n_E))
                new_T = (new_e, r, ta) if self.rpc_h(r) else (h, r, new_e)
                if new_T not in self.pool:
                    T_negs.append((new_T[0], new_T[2]))
                    break
        return np.array(T_negs)


    def em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
             
        t0 = time.time()
        print('     MR    MRR  TOP1  TOP5  TOP10 TOP100 TIME  (Test)\n   ',
              end = '')
        lp_out = self.link_prediction(sess, 'test')
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
        result = myjson(self.out_dir + 'result')        
        result.update(lp_out)
        myjson(self.out_dir + 'result', result)
                
        if self.lambda_c:
            C_table = sess.run(self.C_table)
            mypickle(self.out_dir + '_C_table', C_table)  
        if self.lambda_d:
            D_table = sess.run(self.D_table)
            mypickle(self.out_dir + '_D_table', D_table)
        
    
    def link_prediction(self, sess, key):   
        """
        Linking Prediction of knowledge graph embedding.
        Return entity MR, MRR, TOP1, TOP5, TOP10, TOP100
        
        Args:
            sess: tf.Session
            key: 'dev' or 'test'
        """
        
        T_poss = self.get_batches(key)
        rank = []
        for T_pos in T_poss:
            lp_h, lp_t = sess.run([self.lp_h, self.lp_t], {self.T_pos: T_pos})
            for i in range(len(T_pos)):
                rank.extend([self.cal_ranks(lp_h[i], list(T_pos[i]), 0), 
                             self.cal_ranks(lp_t[i], list(T_pos[i]), 2)]) 
        
        if key == 'test':
            with open(self.out_dir + 'rank.txt', 'w') as file:
                for x in rank:
                    file.writelines(str(x) + '\n')
            
        rank = np.array(rank)
        MR = round(np.mean(rank), 1)
        MRR = round(np.mean([1 / x for x in rank]), 5)
        TOP1 = round(np.mean(rank == 1), 5)
        TOP5 = round(np.mean(rank <= 5), 5)
        TOP10 = round(np.mean(rank <= 10), 5)
        TOP100 = round(np.mean(rank <= 100), 5)
        
        print('{:>6.1f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f}'. \
              format(MR, MRR, TOP1, TOP5, TOP10, TOP100), end = '')
        
        return {'MR': MR, 'MRR': MRR, 'TOP1': TOP1,
                'TOP5': TOP5, 'TOP10': TOP10, 'TOP100': TOP100}
    
    
    def cal_ranks(self, score, T, idx):
        """
        Cal link prediction rank for a single triple.
        
        Args:
            score: replace an entity (a relation) by all the entity of an real
            triple, shape of [n_E, 3]
            T: raw triple
            idx: the replace place of the triple
        """
        
        rank = np.argsort(score)
        g = self.R_reverse_dict[T[1]].split('-')[idx // 2]
        valid_e = self.G_dict[g]
        left, right = valid_e[0], valid_e[-1]
        out = 1 
        for x in rank:
            if left <= x <= right:
                if x == T[idx]:
                    break
                else:
                    new_T = T.copy()
                    new_T[idx] = x
                    if tuple(new_T) not in self.pool:
                        out += 1
        return out  


    def initialize_variables(self, mode):
        """
        Initialize and display variables and shapes.
        
        Args:
            mode: 'train' or 'predict'
        """
        
        tvs = collections.OrderedDict()
        for v in tf.trainable_variables():
            name = re.match('^(.*):\\d+$', v.name).group(1)
            shape = v.shape.as_list()
            tvs[name] = shape
                
        if mode == 'train':
            if self.lambda_c or self.lambda_d:
                p = self.data_dir + self.model + '/'
                kpi, file = 0, ''
                for _ in os.listdir(p):
                    if '(C 0.0, D 0.0)' in _:
                        tmp = myjson(p + _ + '/result')['TOP100']
                        if tmp > kpi:
                            kpi = tmp
                            file = _
                p += (file + '/')
            else:
                p = None
        else:
            p = self.out_dir
        
        if p:
            p += 'model.ckpt'
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
        else:
            ivs = {}
                                
        if mode == 'train' or (mode != 'train' and not self.do_train):
            for v, shape in tvs.items():
                print('    {}{} : {}'.format('*' if v in ivs else '-', v,
                                             shape))
            print()
        

    def em_evaluate(self, sess):
        """
        Cal scores for 25 relations' all possible triplets.
        
        Args:
            sess: tf.Session
        """

        _dir = self.out_dir + 'scores/'
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        G = ['DI', 'DR', 'GP', 'PH', 'SM']
        for X in G:
            idx_X = self.G_dict[X]
            n_X = len(idx_X) 
            for Y in G:
                idx_Y = self.G_dict[Y]
                n_Y = len(idx_Y)                
                r = self.R_dict[X + '-' + Y]
                t0 = time.time()
                S = np.zeros((n_X, n_Y))
                for i in range(n_X):
                    h = idx_X[i] 
                    T = np.array([[h, r, ta] for ta in idx_Y])
                    S[i, :] = sess.run(self.score_pos, {self.T_pos: T})
                mypickle('{}{}-{}'.format(_dir, X, Y), S)
                print('>>  {}-{} : {:5} * {:5} = {:9} ({:.2f} min)'.format( \
                       X, Y, n_X, n_Y, n_X * n_Y, (time.time() - t0) / 60))
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        for mode in ['train', 'predict', 'evaluate']:
            if eval('self.do_' + mode):
                print('\n>>  {} Process.'.format(mode.title()))
                self.initialize_variables(mode)        
                with tf.Session(config = config) as _:
                    tf.global_variables_initializer().run()   
                    exec('self.em_' + mode + '(_)')
                


def myjson(p, data = None):
    """
    Read (data is None) or Dump (data is not None) a json file.    
    
    Args:
        p: file path
        data(None): json data
    """

    if '.json' not in p:
        p += '.json'
    if data is None:
        with open(p) as file:
            return json.load(file)
    else:
        with open(p, 'w') as file:
            json.dump(data, file) 
            

def mypickle(p, data = None):
    """
    Read (data is None) or Dump (data is not None) a pickle file.    
    
    Args:
        p: file path
        data(None): pickle data
    """

    if '.data' not in p:
        p += '.data'
    if data is None:
        with open(p, 'rb') as file:
            return pickle.load(file)
    else:
        with open(p, 'wb') as file:
            pickle.dump(data, file)  