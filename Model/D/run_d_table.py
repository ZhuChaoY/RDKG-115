import argparse
from train_d_table import D_Table


parser = argparse.ArgumentParser(description = 'Run D Table')
#'bert', 'biobert', 'scibert', 'pubmedbert'
parser.add_argument('--model', type = str, default = 'pubmedbert',
                    help = 'model name') 
parser.add_argument('--len_d', type = int, default = 150,
                    help = 'length of the text') 
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
obj = D_Table(args)
if args.do_train:
    obj._train()
if args.do_predict:
    obj._predict()