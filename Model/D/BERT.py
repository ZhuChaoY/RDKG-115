import math
import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, config, n_label = None):
        super(BERT, self).__init__()        
        self.n_label = n_label
        self.n_token = config['vocab_size']
        self.n_position = config['max_position_embeddings']
        self.n_segment = config['type_vocab_size']
        self.H = config['hidden_size']
        self.n_head = config['num_attention_heads']
        self.d_r = config['hidden_dropout_prob']
        self.inter_size = config['intermediate_size']
        self.n_layer = config['num_hidden_layers']
            
        self.token_table = nn.Embedding(self.n_token, self.H, padding_idx = 0)
        self.position_table = nn.Embedding(self.n_position, self.H)
        self.segment_table = nn.Embedding(self.n_segment, self.H)

        self.layernorm = LayerNorm(self.H)
        self.dropout1 = nn.Dropout(self.d_r)
        self.bert_layers = nn.ModuleList([BERT_Layer(self.H, self.n_head,
                      self.d_r, self.inter_size) for _ in range(self.n_layer)])
        if self.n_label is not None:
            self.pool_fc = nn.Linear(self.H, self.H)
            self.pool_ac = nn.Tanh()
            self.dropout2 = nn.Dropout(self.d_r)
            self.logit_fc = nn.Linear(self.H, n_label)
        
        
    def forward(self, token, segment = None, mode = 'train'):
        #(B, L, H) + [B, L, H] + [B, L, H] ==> [B, L, H]
        if segment is None:
            segment = torch.zeros_like(token)
        position = torch.arange(token.size(1), dtype = torch.long,
                    device = token.device).unsqueeze(0).expand_as(token)

        em_out = self.token_table(token) + \
                  self.position_table(position) + \
                  self.segment_table(segment)
        em_out = self.layernorm(em_out)
        if mode == 'train':
            em_out = self.dropout1(em_out)
        
        #[B, 1, 1, l_seq]
        att_mask = (token != 0).long().unsqueeze(1).unsqueeze(2)
        att_mask = att_mask.to(dtype = next(self.parameters()).dtype)
        
        #[B, L, H] ==> [B, L, H]
        sequence_out = [em_out]
        for bert_layer in self.bert_layers:
            sequence_out.append(bert_layer(sequence_out[-1], att_mask, mode))
        
        if self.n_label is not None:        
            #(B, L, H) ==> (B, H) ==> (B, n_label)
            pooled_out = self.pool_ac(self.pool_fc(sequence_out[-1][:, 0]))
            if mode == 'train':
                pooled_out = self.dropout2(pooled_out)
            logits = self.logit_fc(pooled_out)
            return logits
        else:
            sentence_out = torch.mean(sequence_out[-2], 1)
            return sentence_out
        
    
    def load_checkpoint(self, checkpoint, mode = 'pretrain'):
        old_dict = torch.load(checkpoint, map_location = 'cpu')
        old_keys = list(old_dict.keys())
        
        if mode == 'pretrain':
            k = 199 if self.n_label is not None else 197
            old_keys = old_keys[: k]
            new_keys = [x for x, y in self.named_parameters()][: k]                       
            new_dict = \
                {new: old_dict[old] for old, new in zip(old_keys, new_keys)}
        elif mode == 'predict':
            new_dict = old_dict
        self.load_state_dict(new_dict, strict = False)
        print('    (Load checkpoint from: {})'.format(checkpoint))



class BERT_Layer(nn.Module):
    def __init__(self, H, n_head, d_r, inter_size):
        super(BERT_Layer, self).__init__()
        self.MHA = MHA_Layer(H, n_head, d_r)
        self.FFN = FFN_Layer(H, inter_size, d_r)
        
        
    def forward(self, pre_out, att_mask, mode = 'train'):
        #[B, L, H] ==> [B, L, H]
        att_out = self.MHA(pre_out, att_mask, mode)
        ffn_out = self.FFN(att_out, mode)
        return ffn_out
        
    

class MHA_Layer(nn.Module):
    def __init__(self, H, n_head, d_r):
        super(MHA_Layer, self).__init__()
        self.H = H
        self.n_head = n_head
        self.head_size = int(H / n_head)

        self.Q_table = nn.Linear(H, H)
        self.K_table = nn.Linear(H, H)
        self.V_table = nn.Linear(H, H)
        self.dropout1 = nn.Dropout(d_r)
        self.fc = nn.Linear(H, H)
        self.dropout2 = nn.Dropout(d_r)
        self.layernorm = LayerNorm(H)


    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, pre_out, att_mask, mode = 'train'):
        #[B, L, H] ==> [B, n_head, L, head_size]
        Q = self.transpose(self.Q_table(pre_out))
        K = self.transpose(self.K_table(pre_out))
        V = self.transpose(self.V_table(pre_out))

        #[B, n_head, L, head_size] * [B, n_head, head_size, L] ==>
        #[B, n_head, L, l_seq]
        att_score = torch.matmul(Q, K.transpose(-1, -2)) / \
                    math.sqrt(self.head_size) + (1.0 - att_mask) * -10000.0
        att_prob = nn.Softmax(dim = -1)(att_score)
        if mode == 'train':
            att_prob = self.dropout1(att_prob)
        
        #[B, n_head, L, L] * [B, n_head, L, head_size] ==>
        #[B, n_head, L, head_size] ==> [B, L, n_head, head_size] ==>
        #[B, L, H] ==> [B, L, H]
        att_out = torch.matmul(att_prob, V).permute(0, 2, 1, 3).contiguous()
        att_out = self.fc(att_out.view(*att_out.size()[:-2] + (self.H, )))
        if mode == 'train':
            att_out = self.dropout2(att_out)
        att_out = self.layernorm(pre_out + att_out)
        return att_out



class FFN_Layer(nn.Module):
    def __init__(self, H, inter_size, d_r):
        super(FFN_Layer, self).__init__()

        self.fc1 = nn.Linear(H, inter_size)
        self.dropout1 = nn.Dropout(d_r)
        self.fc2 = nn.Linear(inter_size, H)
        self.dropout2 = nn.Dropout(d_r)
        self.layernorm = LayerNorm(H)
 
    
    def forward(self, att_out, mode = 'train'):
        #[B, L, H] ==> [B, L, inter_size] ==>[B, L, H]
        ffn_out = gelu(self.fc1(att_out))
        if mode == 'train':
            ffn_out = self.dropout1(ffn_out)
        ffn_out = self.fc2(ffn_out)
        if mode == 'train':
            ffn_out = self.dropout2(ffn_out)
        ffn_out = self.layernorm(ffn_out + att_out)
        return ffn_out
    
    
    
class LayerNorm(nn.Module):
    def __init__(self, hidden, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.eps = eps


    def forward(self, x):
        u = x.mean(-1, keepdim = True)
        s = (x - u).pow(2).mean(-1, keepdim = True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))