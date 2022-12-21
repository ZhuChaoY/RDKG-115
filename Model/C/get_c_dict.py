import os
import sys
import random
import pandas as pd
sys.path.append('../')
from train_kge import mypickle, myjson


p = 'C_dict.data'
if os.path.exists(p):
    C_dict = mypickle(p)
    print('>>  C_dict already exists !')
else:
    E = list(pd.read_csv('../KG/entity.csv')['E']) 
    E_dict = myjson('../../Annotation/E_dict')
        
    c_count, C = {}, []
    for e in E:
        c1, c2, c3 = E_dict[e]['C']
        for c in c2 + c3:
            if c not in c_count:
                c_count[c] = 1
            else:
                c_count[c] += 1
        C.append([c1, c2, c3])

    c_list = \
        ['*DI', '*DR', '*GP', '*PH', '*SM'] + \
        [key for key, value in c_count.items() if value >= 10]
    c_dict = dict(zip(sorted(c_list), range(len(c_list))))     
    num_c = 32

    C_dict = []
    for c1, c2, c3 in C:
        c1 = [c_dict[c1[0]]]
        c2 = [c_dict[c] for c in c2 if c in c_dict]
        c3 = [c_dict[c] for c in c3 if c in c_dict]

        if 1 + len(c2) >= num_c:
            c = c1 + random.sample(c2, num_c - 1)
        elif 1 + len(c2 + c3) >= num_c:
            c = c1 + c2 + random.sample(c3, num_c - len(c2) - 1)
        else:
            c = (c1 + c2 + c3) * (num_c // (1 + len(c2 +c3)))
            c += c1 * (num_c - len(c))
        C_dict.append(sorted(c))

    mypickle(p, C_dict)
    print('>>  Generate C_dict Done !')