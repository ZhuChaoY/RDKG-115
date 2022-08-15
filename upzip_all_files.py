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
        

for path in ['Dataset/DR', 'Dataset/PC', 'Dataset/PK', 'Dataset/RD',
             'Dataset/SD', 'Dataset/train', 'Dataset/triplet',
             'Annotation/E_dict']:
    unzip_file(path)

    