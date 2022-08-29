import os
import argparse
import tensorflow as tf
from Models import *
    

parser = argparse.ArgumentParser(description = 'KGE')

#'TransE', 'TransH',  'RotatE'
parser.add_argument('--model', type = str, default = 'RotatE',
                    help = 'model name') 
parser.add_argument('--dim', type = int, default = 100,
                    help = 'embedding dimension')
parser.add_argument('--margin', type = float, default = 2.5,
                    help = 'margin value')
parser.add_argument('--lambda_c', type = float, default = 0.0,
                    help = 'lambda for category')
parser.add_argument('--lambda_d', type = float, default = 0.0, 
                    help = 'lambda for description')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'l2 penalty coefficient')
parser.add_argument('--l_r', type = float, default = 5e-3, 
                    help = 'learning rate')
parser.add_argument('--train_batch_size', type = int, default = 25000,
                    help = 'batch size for SGD')
parser.add_argument('--eval_batch_size', type = int, default = 30,
                    help = 'batch size for link prediction')
parser.add_argument('--epoches', type = int, default = 800,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 2,
                    help = 'earlystop steps')
parser.add_argument('--do_train', type = int, default = 1,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = int, default = 1,
                    help = 'whether to predict')
parser.add_argument('--do_evaluate', type = int, default = 0,
                    help = 'whether to evaluate')
parser.add_argument('--gpu', type = str, default = '0',
                    help = 'gpu number')
    
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
             
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
model = eval(args.model + '(args)')
model.run(config) 