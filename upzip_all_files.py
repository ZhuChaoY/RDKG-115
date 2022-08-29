import os
import zipfile


def unzip_file(p):
    print('>>    {} : '.format(p), end = '')
    if '.zip' not in p:
        p += '.zip'
    if not os.path.exists(p):
        print('Already upzipped !')
    else:        
        F = zipfile.ZipFile(p, 'r')
        for f in F.namelist():
            F.extract(f, '/'.join(p.split('/')[: -1]))
        F.close()
        
        os.remove(p)
        
        print('Done !')
        

for path in ['Annotation/E_dict', 'Model/KG/train',
             'RDKG-115/RDKG-115', 'RDKG-115/RDKG-115 (plus inferred)']:
    unzip_file(path)