import argparse
from train_kge import KGE


parser = argparse.ArgumentParser(description = 'Run KGE')

#'TransE', 'TransH',  'ConvKB', 'RotatE'
parser.add_argument('--model', type = str, default = 'TransE',
                    help = 'model name') 
parser.add_argument('--dim', type = int, default = 128,
                    help = 'embedding dimension')
parser.add_argument('--margin', type = float, default = 1.0,
                    help = 'margin value')
parser.add_argument('--lambda_c', type = float, default = 0.0,
                    help = 'lambda for category')
parser.add_argument('--lambda_d', type = float, default = 0.0, 
                    help = 'lambda for description')
parser.add_argument('--l_r', type = float, default = 5e-4, 
                    help = 'learning rate')
parser.add_argument('--l2', type = float, default = 1e-4,
                    help = 'l2 penalty coefficient')
parser.add_argument('--train_n_batch', type = int, default = 200,
                    help = 'number of batch for SGD')
parser.add_argument('--eval_n_sample', type = int, default = 10000,
                    help = 'number of sample for link prediction eval')
parser.add_argument('--eval_batch_size', type = int, default = 25,
                    help = 'batch size for link prediction eval, 1 for ConvKB')
parser.add_argument('--epoches', type = int, default = 200,
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
model = KGE(args)
model.run()