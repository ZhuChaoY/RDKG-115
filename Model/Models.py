import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    
    def __init__(self, dim, lambda_c, lambda_d, margin, l2, n_E, n_R, device):
        super().__init__()
        self.dim = dim
        self.lambda_c = lambda_c
        self.lambda_d = lambda_d
        self.margin = margin
        self.l2 = l2
        self.n_E = n_E
        self.n_R = n_R
        self.device = device
        
        self.E_table = self.get_embedding(self.n_E)
        self.R_table = self.get_embedding(self.n_R)
        
        if self.lambda_c:
            self.C_layer()
        if self.lambda_d:
            self.D_layer()


    def C_layer(self):
        with open('C/C_dict.data', 'rb') as file:
            C_dict = pickle.load(file)
        c_list = []
        for c in C_dict:
            c_list.extend(c)
        n_C = len(set(c_list))
        self.C_dict = torch.tensor(C_dict).long().to(self.device)
        k = 1 if self.__class__.__name__ != 'RotatE' else 2
        self.raw_C_table = self.get_embedding(n_C, k * self.dim)

    
    def D_layer(self):
        with open('D/pubmedbert/D_table.data', 'rb') as file:
            init_D = torch.tensor(pickle.load(file)).to(self.device)
        H = list(init_D.shape)[1]
        self.raw_D_table = nn.Embedding(self.n_E, H)
        self.raw_D_table.weight = nn.Parameter(init_D)
        k = 1 if self.__class__.__name__ != 'RotatE' else 2
        self.d_fc = nn.Linear(H, k * self.dim)
        
    
    def get_embedding(self, n, dim = None, K = None):
        if not dim:
            dim = self.dim
        if not K:
            K = np.sqrt(6.0 / dim)
        table = nn.Embedding(n, dim)
        nn.init.uniform_(table.weight.data, -K, K)
        return table.to(self.device)
    
        
    def cal_l2_loss(self, vs):
        if self.l2 == 0.0:
            return 0.0
        else:
            l2_loss = 0.0
            n = len(vs)
            for v in vs:
                l2_loss = l2_loss + torch.mean(v ** 2)
            return l2_loss / n    


    def normalize(self, v):
        if type(v) == list:
            return [F.normalize(x, 2, -1) for x in v]
        else:
            return F.normalize(v, 2, -1)
        
    
    def get_c_d_table(self, T, mode, idx):
        C_table, h_c, t_c = None, None, None
        if self.lambda_c:
            c_table = self.raw_C_table(self.C_dict)
            C_table = self.normalize(torch.mean(c_table, 1))
            h_c, t_c = C_table[T[:, 0]], C_table[T[:, 2]]
            if mode in ['eval_h', 'eval_t']:
                C_table = C_table[idx]
        
        D_table, h_d, t_d = None, None, None
        if self.lambda_d:
            all_E = torch.tensor(range(self.n_E)).to(self.device)
            d_table = self.raw_D_table(all_E)
            D_table = self.normalize(self.d_fc(d_table))
            h_d, t_d = D_table[T[:, 0]], D_table[T[:, 2]]
            if mode in ['eval_h', 'eval_t']:
                D_table = D_table[idx]
        
        return C_table, h_c, t_c, D_table, h_d, t_d
    
    
    def project(self, s, h_c, t_c, h_d, t_d, mode):
        s_c = None
        if h_c is not None:
            if mode == 'eval_h':
                t_c = torch.unsqueeze(t_c, 1)
            elif mode == 'eval_t':
                h_c = torch.unsqueeze(h_c, 1)
            s_c = self._project(s, h_c - t_c)

        s_d = None
        if h_d is not None:
            if mode == 'eval_h':
                t_d = torch.unsqueeze(t_d, 1)
            elif mode == 'eval_t':
                h_d = torch.unsqueeze(h_d, 1)
            s_d = self._project(s, h_d - t_d)
                
        return s_c, s_d
        
        
    def _project(self, s, p):        
        s = self.normalize(s)
        return p - torch.sum(s * p, -1, True) * s  
    
        
    def get_score(self, s, s_c, s_d):
        score_s, score_c, score_d = self._cal_score(s), None, None
        score = 100 - score_s
        if s_c is not None:
            score_c = self._cal_score(s_c)
            score = score + self.lambda_c * score_c
        if s_d is not None:
            score_d = self._cal_score(s_d)
            score = score + self.lambda_d * score_d
        return score, score_s, score_c, score_d
    
    
    def _cal_score(self, s):
        return torch.norm(s, 1, -1)
    
    
    def cal_loss(self, T_pos, T_neg):
        score_pos, l2_loss_pos = self.cal_score(T_pos, 'train')
        score_neg, l2_loss_neg = self.cal_score(T_neg, 'train')
        loss = torch.mean(F.relu(self.margin - score_pos + score_neg)) + \
               self.l2 * (l2_loss_pos + l2_loss_neg) / 2
        return loss
    
    
    def load_checkpoint(self, checkpoint):
        dic = torch.load(checkpoint, map_location = 'cpu')
        self.load_state_dict(dic, strict = False)
        print('    (Load checkpoint from: {})'.format(checkpoint))
        
    

class TransE(BaseModel):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, *args):
        super().__init__(*args)
        
        
    def cal_score(self, T, mode, idx = None):
        T = T.to(self.device)
        h = self.E_table(T[:, 0])
        r = self.R_table(T[:, 1])
        t = self.E_table(T[:, 2])
        
        if mode == 'train':
            l2_loss = self.cal_l2_loss([h, r, t])
        
        h, r, t = self.normalize([h, r, t])     
        C_table, h_c, t_c, D_table, h_d, t_d = self.get_c_d_table(T, mode, idx)

        if mode in ['train', 'eval']:
            s = h + r - t
            s_c, s_d = self.project(s, h_c, t_c, h_d, t_d, mode)
        else:
            E = self.E_table(idx)
            E = self.normalize(E)
            if mode == 'eval_h':
                s = E + torch.unsqueeze(r - t, 1)
                s_c, s_d = self.project(s, C_table, t_c, D_table, t_d, mode)
            elif mode == 'eval_t':
                s = torch.unsqueeze(h + r, 1) - E
                s_c, s_d = self.project(s, h_c, C_table, h_d, D_table, mode)
                
        score, score_s, score_c, score_d = self.get_score(s, s_c, s_d)
        
        if mode == 'train':
            return score, l2_loss
        elif mode == 'eval':
            return score, score_s, score_c, score_d
        else:
            return score
        


class TransH(BaseModel):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, *args):
        super().__init__(*args)
        self.P_table = self.get_embedding(self.n_R)
        
    
    def transfer(self, e, p):
        return e - torch.sum(e * p, -1, True) * p
    
    
    def cal_score(self, T, mode, idx = None):
        T = T.to(self.device)
        h = self.E_table(T[:, 0])
        r = self.R_table(T[:, 1])
        t = self.E_table(T[:, 2])
        p = self.P_table(T[:, 1])
        
        if mode == 'train':
            l2_loss = self.cal_l2_loss([h, r, t, p])
                
        h, r, t, p = self.normalize([h, r, t, p])
        h = self.transfer(h, p)
        t = self.transfer(t, p)
        C_table, h_c, t_c, D_table, h_d, t_d = self.get_c_d_table(T, mode, idx)
        
        if mode in ['train', 'eval']:
            s = h + r - t
            s_c, s_d = self.project(s, h_c, t_c, h_d, t_d, mode)
        else:
            E = self.E_table(idx)
            E = self.transfer(self.normalize(E), torch.unsqueeze(p, 1))
            if mode == 'eval_h':
                s = E + torch.unsqueeze(r - t, 1)
                s_c, s_d = self.project(s, C_table, t_c, D_table, t_d, mode)
            elif mode == 'eval_t':
                s = torch.unsqueeze(h + r, 1) - E
                s_c, s_d = self.project(s, h_c, C_table, h_d, D_table, mode)
                
        score, score_s, score_c, score_d = self.get_score(s, s_c, s_d)
        
        if mode == 'train':
            return score, l2_loss
        elif mode == 'eval':
            return score, score_s, score_c, score_d
        else:
            return score
        
     
class ConvKB(BaseModel):
    """
    A Novel Embedding Model for Knowledge Base Completion Based on 
    Convolutional Neural Network.
    """
    
    def __init__(self, *args):
        super().__init__(*args)
        self.n_filter = 50
        init_f = nn.init.normal_(torch.zeros(self.n_filter, 1, 1, 3),
                                 0.5, 0.05).float().to(self.device)
        init_f[:, :, :, -1] *= -1        
        self.kernel = nn.Parameter(init_f)
        self.act = nn.ReLU()
        self.fc = nn.Linear(self.dim * self.n_filter, 1, bias = False)
        
        
    def project(self, s, h_c, t_c, h_d, t_d, mode):
        s_c = None
        if h_c is not None:
            s_c = self._project(s, h_c - t_c)

        s_d = None
        if h_d is not None:
            s_d = self._project(s, h_d - t_d)
                
        return s_c, s_d
        
        
    def _project(self, s, p):        
        s = F.normalize(s, 2, 2)
        return p - torch.sum(s * p, 2, True) * s      


    def get_score(self, s, s_c, s_d, mode):
        score_s, score_c, score_d = self._cal_score(s), None, None
        if mode in ['eval_h', 'eval_t']:
            score_s = torch.unsqueeze(score_s, 0)
        score = -score_s
        if s_c is not None:
            score_c = self._cal_score(s_c)
            if mode in ['eval_h', 'eval_t']:
                score_c = torch.unsqueeze(score_c, 0)
            score = score + self.lambda_c * score_c
        if s_d is not None:
            score_d = self._cal_score(s_d)
            if mode in ['eval_h', 'eval_t']:
                score_d = torch.unsqueeze(score_d, 0)
            score = score + self.lambda_d * score_d
        return score, score_s, score_c, score_d


    def _cal_score(self, s):
        #(B, F, D, 1) ==> (B, D * F) ==> (B, 1) ==> (B, )
        return torch.squeeze(self.fc(self.act(s).view(-1,
               self.dim * self.n_filter)), -1)
        

    def cal_score(self, T, mode, idx = None):
        T = T.to(self.device)
        h = self.E_table(T[:, 0])
        r = self.R_table(T[:, 1])
        t = self.E_table(T[:, 2])
        
        if mode == 'train':
            l2_loss = self.cal_l2_loss([h, r, t])
        
        h, r, t = self.normalize([h, r, t])    
        C_table, h_c, t_c, D_table, h_d, t_d = self.get_c_d_table(T, mode, idx)
        h = h.view(-1, 1, self.dim, 1)
        r = r.view(-1, 1, self.dim, 1)
        t = t.view(-1, 1, self.dim, 1)
        if C_table is not None:
            C_table = C_table.view(-1, 1, self.dim, 1)
            h_c = h_c.view(-1, 1, self.dim, 1)
            t_c = t_c.view(-1, 1, self.dim, 1)
        if D_table is not None:
            D_table = D_table.view(-1, 1, self.dim, 1)
            h_d = h_d.view(-1, 1, self.dim, 1)
            t_d = t_d.view(-1, 1, self.dim, 1)
        
        if mode in ['train', 'eval']:
            #(B, 1, D, 3) ==> (B, F, D, 1) 
            s = F.conv2d(torch.cat([h, r, t], -1), self.kernel)
            #(B, F, D, 1) * (B, 1, D, 1) ==> (B, F, D, 1)
            s_c, s_d = self.project(s, h_c, t_c, h_d, t_d, mode)
        else:                           
            E = self.E_table(idx)
            n_E = list(E.shape)[0]
            E = self.normalize(E).view(-1, 1, self.dim, 1)
            #(1, 1, D, 2) * (n_E, ) ==> (n_E, 1, D, 2)
            #(n_E, 1, D, 3) ==> (n_E, F, D, 1)
            #(n_E, F, D, 1) * (n_E, 1, D, 1) ==> (n_E, F, D, 1)
            if mode == 'eval_h':                
                s = F.conv2d(torch.cat([E, torch.cat([r, t], -1). \
                    repeat((n_E, 1, 1, 1))], -1), self.kernel)
                s_c, s_d = self.project(s, C_table, t_c, D_table, t_d, mode)
            elif mode == 'eval_t':
                s = F.conv2d(torch.cat([torch.cat([h, r], -1). \
                    repeat((n_E, 1, 1, 1)), E], -1), self.kernel)
                s_c, s_d = self.project(s, h_c, C_table, h_d, D_table, mode)
                
        score, score_s, score_c, score_d = self.get_score(s, s_c, s_d, mode)
        
        if mode == 'train':
            return score, l2_loss
        elif mode == 'eval':
            return score, score_s, score_c, score_d
        else:
            return score
        
      

class RotatE(BaseModel):
    """
    ROTATE: KNOWLEDGE GRAPH EMBEDDING BY RELATIONAL ROTATION IN COMPLEX SPACE.
    """
    
    def __init__(self, *args):
        super().__init__(*args)
        self.R_table = self.get_embedding(self.n_R, None, 3.1415926)
        self.E_I_table = self.get_embedding(self.n_E)
            
        
    def cal_score(self, T, mode, idx = None):
        T = T.to(self.device)
        h_r = self.E_table(T[:, 0])
        t_r = self.E_table(T[:, 2])
        h_i = self.E_I_table(T[:, 0])
        t_i = self.E_I_table(T[:, 2])
        r = self.R_table(T[:, 1])
        r_r, r_i = torch.cos(r), torch.sin(r)

        if mode == 'train':
            l2_loss = self.cal_l2_loss([h_r, t_r, h_i, t_i])
            
        h_r, t_r, h_i, t_i = self.normalize([h_r, t_r, h_i, t_i])
        C_table, h_c, t_c, D_table, h_d, t_d = self.get_c_d_table(T, mode, idx)
        
        if mode in ['train', 'eval']:
            re = h_r * r_r - h_i * r_i - t_r
            im = h_r * r_i + h_i * r_r - t_i
            s = torch.cat([re, im], -1)
            s_c, s_d = self.project(s, h_c, t_c, h_d, t_d, mode)
        else:
            E = self.E_table(idx)
            E_I = self.E_I_table(idx)
            E, E_I = self.normalize([E, E_I])
            if mode == 'eval_h':
                re = E * torch.unsqueeze(r_r, 1) - \
                     E_I * torch.unsqueeze(r_i, 1) - torch.unsqueeze(t_r, 1)
                im = E * torch.unsqueeze(r_i, 1) + \
                     E_I * torch.unsqueeze(r_r, 1) - torch.unsqueeze(t_i, 1)
                s = torch.cat([re, im], -1)
                s_c, s_d = self.project(s, C_table, t_c, D_table, t_d, mode)
            elif mode == 'eval_t':
                re = torch.unsqueeze(h_r * r_r - h_i * r_i, 1) - E
                im = torch.unsqueeze(h_r * r_i + h_i * r_r, 1) - E_I
                s = torch.cat([re, im], -1)
                s_c, s_d = self.project(s, h_c, C_table, h_d, D_table, mode) 
                
        score, score_s, score_c, score_d = self.get_score(s, s_c, s_d)
        
        if mode == 'train':
            return score, l2_loss
        elif mode == 'eval':
            return score, score_s, score_c, score_d
        else:
            return score