# RDKG-115
Assisting Drug Repurposing and Discovery for Rare Diseases by Trimodal Knowledge Graph Embedding   

## Abstract
Rare diseases (RDs) individually affect a small number of people but collectively bother a crowd worldwide. They are challenging to diagnose accurately and lack drugs, also known as orphan diseases. Frustrated with expensive and inefficient drug development for rare diseases, pharmaceutical companies favor drug repurposing from existing drugs for other diseases. Compared with traditional approaches to drug repurposing, such as experimental and computational methods, knowledge graph embedding (KGE) based approaches are more convenient and efficient. By sorting the candidate drug set based on a defined KGE scoring function, drug repurposing is equivalent to a link prediction task. Furthermore, KGE models can replenish existing knowledge by adding multimodal information from more sources, such as categories and descriptions of entities. Here, we constructed RDKG-115, a trimodal rare disease knowledge graph involving 115 RDs, composed of around 4.4 million refined triplets, 34,401 entities and 25 relations, based on 372,384 high-quality literature and three biomedical datasets: DRKG, Pathway Commons and PharmKG. After that, we built a trimodal KGE model by reverse-hyperplane projection, and inferred 4,196 reliable new inferred triplets from RDKG-115. We wise that RDKG-115 could provide potential meaningful clues for researchers, facilitating drug repurposing and discovery for RDs.  

## Files
### Annotation/
**E_dict.json** : json file of standard entirt set  

### Model/ 
**KGE.py** : Class of processing and tool functions for Knowledge Graph Embedding    
**Models.py** : TransE, TransH, RotatE structure    
**Run_KGE.py** : Run KGE.py        
#### C&D/
**C_dict.data** : Dict of entity category annotation  
**D_table.data** : Table of entity description annotation  
**E_index.json** : Entity index dict for C_dict and D_table  
**get_C_dict.py** : Run it to get C_dict.data and E_index.data    
**D_Table.py** : Structure for training description table        
**Optimization.py** : Training optimization of BioBERT     
**Tokenization.py** : Tokenization function of BioBERT     
**Run_D_Table.py** : Run it to get D_table.data  
#### KG/
**entity.csv** : Entities of RDKG-115  
**relation.csv** : Relations of RDKG-115  
**train.csv** : Train set of RDKG-115 (98%)    
**dev.csv** : Dev set of RDKG-115 (1%)  
**test.csv** : Test set of RDKG-115 (1%)  
#### Pretrained BioBERT/  
**biobert_config.json**  
**biobert_model.ckpt.data-00000-of-00001**  
**biobert_model.ckpt.index**  
**biobert_model.ckpt.meta**  
**vocab.txt**  
Self download biobert_model.ckpt from https://github.com/dmis-lab/biobert     

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)   
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://arxiv.org/pdf/1712.02121.pdf)   
(4) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)    
(5) **BioBERT**: [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746v2.pdf)

## Version
(1) python 3.6  
(2) tensorflow-gpu 1.12.0  
(3) numpy 1.17.4  

## Operating Instructions
(1) Run get_E_dict.py to get E_dict.json in **Annotation/**     
```
python get_E_dict.py
```

(2) Run get_C_dict.py to get C_dict.data and E_index.data in **Model/C&D/** (Already in the folder, you can not run)    
```
python get_C_dict.py   
```

(3) Run Run_D_Table.py to get D_table.data in **Model/C&D/**     
```
python Run_D_Table.py --len_d 150 --dim 200 --l_r 1e-5 --batch_size 8 --epoches 10 --earlystop 1   
```

(4) Run Run_KGE.py to train TransE, TransH, and ConvKB in **Model/**
#### Parameter Interpretation  
lanta_c == 0 and lanta_d == 0 : S  
lanta_c != 0 and lanta_d == 0 : S + C  
lanta_c == 0 and lanta_d != 0 : S + D  
lanta_c != 0 and lanta_d != 0 : S + C + D  

**[disease]** from the abbreviation of disease names as follow      
{'ald' : 'alzheimer_disease',  
 'coc' : 'colon_cancer',  
 'cop' : 'copd',  
 'chd' : 'coronary_heart_disease',  
 'dia' : 'diabetes',  
 'gac' : 'gallbladder_cancer',  
 'gsc' : 'gastric_cancer',  
 'hef' : 'heart_failure',  
 'lic' : 'liver_cancer',  
 'luc' : 'lung_cancer',  
 'rha' : 'rheumatoid_arthritis',  
 'can' : '_cancer5',  
 'dis' : '_disease11'}   

**TransE**:   
```
python Run_KGE.py --model TransE --disease [disease] --dim 200 --margin 0.6 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 1000
```
**TransH**:  
```
python Run_KGE.py --model TransH --disease [disease] --dim 200 --margin 0.6 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 1000
```
**ConvKB**:  
```
python Run_KGE.py --model ConvKB --disease [disease] --dim 200 --n_filter 10 --lanta_c 0.0 --lanta_d 0.0 --l_r 1e-3 --epoches 200
```

The above are the parameters for S.  
For S + C, S + D, and S + C + D, (l_r = 5e-4,  epoches = 200) is recommended.  
