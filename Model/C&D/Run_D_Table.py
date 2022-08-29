import os
import argparse
import tensorflow as tf
from D_Table import D_Table


parser = argparse.ArgumentParser(description = 'Run_D_table')
parser.add_argument('--len_d', type = int, default = 150,
                    help = 'length of the text') 
parser.add_argument('--dim', type = int, default = 100,
                    help = 'embedding dim')
parser.add_argument('--l_r', type = float, default = 1e-5, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 8,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 5,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 1,
                    help = 'training epoches')
parser.add_argument('--do_train', type = int, default = 1,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = int, default = 1,
                    help = 'whether to predict')
parser.add_argument('--gpu', type = str, default = '0',
                    help = 'gpu number')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
model = D_Table(args)
model.run(config)