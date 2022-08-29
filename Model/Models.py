from KGE import KGE
import tensorflow as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def kge_variables(self):
        pass
    
        
    def em_structure(self, h, r, t, key = 'pos'):
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(s ** 2, -1)
            
    
    def cal_lp_score(self, h, r, t):        
        s_rpc_h = self.E_table + tf.expand_dims(r - t, 1)  
        s_rpc_t = tf.expand_dims(h + r, 1) - self.E_table
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        if self.lambda_c:
            c_rpc_h = self.projector(s_rpc_h, self.C_table - \
                                     tf.expand_dims(self.t_c_pos, 1))
            c_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_c_pos, 1) - self.C_table)
            lp_h -= self.lambda_c * self.cal_score(c_rpc_h)
            lp_t -= self.lambda_c * self.cal_score(c_rpc_t)
        if self.lambda_d:
            d_rpc_h = self.projector(s_rpc_h, self.D_table - \
                                     tf.expand_dims(self.t_d_pos, 1))
            d_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_d_pos, 1) - self.D_table)
            lp_h -= self.lambda_d * self.cal_score(d_rpc_h)
            lp_t -= self.lambda_d * self.cal_score(d_rpc_t)
        return lp_h, lp_t
    


class TransH(KGE):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, args):
        super().__init__(args)
        
            
    def kge_variables(self):
        P_table = tf.get_variable('projection_table', initializer = \
                  tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
        P_table = tf.nn.l2_normalize(P_table, 1)
        self.p = tf.gather(P_table, self.T_pos[:, 1])
        
        self.l2_kge.append(self.p)
        
        
    def em_structure(self, h, r, t, key = 'pos'):       
        self.transfer = lambda s, p: \
            s - tf.reduce_sum(p * s, -1, keepdims = True) * p    
            
        h = self.transfer(h, self.p)
        t = self.transfer(t, self.p)
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(s ** 2, -1)
    
    
    def cal_lp_score(self, h, r, t):        
        p_E_table = self.transfer(self.E_table, tf.expand_dims(self.p, 1))
        s_rpc_h = p_E_table + tf.expand_dims(r - self.transfer(t, self.p), 1)
        s_rpc_t = tf.expand_dims(self.transfer(h, self.p) + r, 1) - p_E_table    
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        if self.lambda_c:
            c_rpc_h = self.projector(s_rpc_h, self.C_table - \
                                     tf.expand_dims(self.t_c_pos, 1))
            c_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_c_pos, 1) - self.C_table)
            lp_h -= self.lambda_c * self.cal_score(c_rpc_h)
            lp_t -= self.lambda_c * self.cal_score(c_rpc_t)
        if self.lambda_d:
            d_rpc_h = self.projector(s_rpc_h, self.D_table - \
                                     tf.expand_dims(self.t_d_pos, 1))
            d_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_d_pos, 1) - self.D_table)
            lp_h -= self.lambda_d * self.cal_score(d_rpc_h)
            lp_t -= self.lambda_d * self.cal_score(d_rpc_t)
        return lp_h, lp_t
    

    
class RotatE(KGE):
    """
    ROTATE: KNOWLEDGE GRAPH EMBEDDING BY RELATIONAL ROTATION IN COMPLEX SPACE.
    """
    
    def __init__(self, args):
        super().__init__(args)
        
            
    def kge_variables(self):
        self.E_I_table = tf.get_variable('entity_imaginary_table',
                         initializer = tf.random_uniform([self.n_E, self.dim],
                                                          -self.K, self.K))
        self.E_I_table = tf.nn.l2_normalize(self.E_I_table, 1)
        self.h_i_pos = tf.gather(self.E_I_table, self.T_pos[:, 0])
        self.t_i_pos = tf.gather(self.E_I_table, self.T_pos[:, -1])
        self.h_i_neg = tf.gather(self.E_I_table, self.T_neg[:, 0])
        self.t_i_neg = tf.gather(self.E_I_table, self.T_neg[:, -1])
        
        self.l2_kge.extend([self.h_i_pos, self.t_i_pos, 
                            self.h_i_neg, self.t_i_neg])
        
        
    def em_structure(self, h, r, t, key):               
        r_r, r_i = tf.cos(r), tf.sin(r)
        if key == 'pos':
            h_i, t_i = self.h_i_pos, self.t_i_pos
        else:
            h_i, t_i = self.h_i_neg, self.t_i_neg
        
        re = h * r_r - h_i * r_i - t
        im = h * r_i + h_i * r_r - t_i
        return tf.concat([re, im], -1)
    
    
    def cal_score(self, s):
        return tf.reduce_sum(tf.abs(s), -1)
    
    
    def cal_lp_score(self, h, r, t):     
        r_r, r_i = tf.cos(r), tf.sin(r)
        s_rpc_h_r = self.E_table * tf.expand_dims(r_r, 1) - self.E_I_table * \
                    tf.expand_dims(r_i, 1) - tf.expand_dims(t, 1)
        s_rpc_h_i = self.E_table * tf.expand_dims(r_i, 1) + self.E_I_table * \
                    tf.expand_dims(r_r, 1) - tf.expand_dims(self.t_i_pos, 1)
        s_rpc_h = tf.concat([s_rpc_h_r, s_rpc_h_i], -1)
        s_rpc_t_r = tf.expand_dims(h * r_r - self.h_i_pos * r_i, 1) - \
                    self.E_table
        s_rpc_t_i = tf.expand_dims(h * r_i + self.h_i_pos * r_r, 1) - \
                    self.E_I_table
        s_rpc_t = tf.concat([s_rpc_t_r, s_rpc_t_i], -1)
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        if self.lambda_c:
            c_rpc_h = self.projector(s_rpc_h, self.C_table - \
                                     tf.expand_dims(self.t_c_pos, 1))
            c_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_c_pos, 1) - self.C_table)
            lp_h -= self.lambda_c * self.cal_score(c_rpc_h)
            lp_t -= self.lambda_c * self.cal_score(c_rpc_t)
        if self.lambda_d:
            d_rpc_h = self.projector(s_rpc_h, self.D_table - \
                                     tf.expand_dims(self.t_d_pos, 1))
            d_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_d_pos, 1) - self.D_table)
            lp_h -= self.lambda_d * self.cal_score(d_rpc_h)
            lp_t -= self.lambda_d * self.cal_score(d_rpc_t)
        return lp_h, lp_t