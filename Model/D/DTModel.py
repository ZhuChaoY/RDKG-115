import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from BERT import BERT


class DTModel(nn.Module):
    
    def __init__(self, config, n_R):
        super().__init__()

        self.bert = BERT(config)
        H = config['hidden_size']
        K = np.sqrt(6.0 / H)
        self.R_table = nn.Embedding(n_R, H)
        nn.init.uniform_(self.R_table.weight.data, -K, K)
            
        
    def forward(self, token, bs = None, r_idx = None, mode = 'train'):
        sentence_out = F.normalize(self.bert(token, None, mode), 2, -1)
        
        if not bs:
            return sentence_out
        else:          
            r = F.normalize(self.R_table(r_idx), 2, -1)
            h_pos = self.transfer(sentence_out[: bs], r)
            t_pos = self.transfer(sentence_out[bs: ], r)         
            h_neg = torch.cat((h_pos[1: ], h_pos[: 1]), 0)
            t_neg = torch.cat((t_pos[1: ], t_pos[: 1]), 0)
            
            loss_pos = (1 - self.cal_simi(h_pos, t_pos)) ** 2
            loss_neg1 = self.cal_simi(h_pos, t_neg) ** 2
            loss_neg2 = self.cal_simi(h_neg, t_pos) ** 2
            
            loss = torch.sum((2 * loss_pos + loss_neg1 + loss_neg2) / 4)
            return loss


    def transfer(self, e, p):
        return e - torch.sum(e * p, -1, True) * p
        
    
    def cal_simi(self, a, b):
        return torch.abs(torch.cosine_similarity(a, b, 1))
        
        
    def load_checkpoint(self, checkpoint, mode = 'pretrain'):
        old_dict = torch.load(checkpoint, map_location = 'cpu')
        old_keys = list(old_dict.keys())
        
        if mode == 'pretrain':
            old_keys = old_keys[: 197]  
            new_keys = [x for x, y in self.named_parameters()][: 197]                       
            new_dict = \
                {new: old_dict[old] for old, new in zip(old_keys, new_keys)}
        elif mode == 'predict':
            new_dict = old_dict
        self.load_state_dict(new_dict, strict = False)
        print('    (Load checkpoint from: {})'.format(checkpoint))