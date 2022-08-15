import os
import json


p = 'E_dict.json'
if os.path.exists(p):
    with open(p) as file:
        E_dict = json.load(file)
    print('>>  E_dict already exists !')
else:
    E_dict = {}
    for k in range(4):
        with open('E_dict_' + str(k) + '.json') as file:
            tmp = json.load(file)
        E_dict.update(tmp)
    with open(p, 'w') as file:
        json.dump(E_dict, file)
    print('>>  Generate E_dict Done !')


n_E = len(E_dict)
g_keys = [x.split('-')[0] for x in E_dict.keys()]
print('\n>>  Totally {} entity in the E_dict.'.format(n_E))
print('       Number Ratio')
for g in ['DI', 'DR', 'GP', 'PH', 'SM']:
    n = g_keys.count(g)
    print('    {:2} {:>6} {:5.3f}'.format(g, n, n / n_E))