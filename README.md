# RDKG-115
[RDKG-115: Assisting drug repurposing and discovery for rare diseases by trimodal knowledge graph embedding](https://www.sciencedirect.com/science/article/pii/S0010482523007278)      

## Cite By
Chaoyu Zhu, Xiaoqiong Xia, Nan Li, Fan Zhong, Zhihao Yang, Lei Liu. RDKG-115: Assisting drug repurposing and discovery for rare diseases by trimodal knowledge graph embedding. Computers in Biology and Medicine, 2023, 107262.

## Abstract
Rare diseases (RDs) may affect individuals in small numbers, but they have a significant impact on a global scale. Accurate diagnosis of RDs is challenging, and there is a severe lack of drugs available for treatment. Pharmaceutical companies have shown a preference for drug repurposing from existing drugs developed for other diseases due to the high investment, high risk, and long cycle involved in RD drug development. Compared to traditional approaches, knowledge graph embedding (KGE) based methods are more efficient and convenient, as they treat drug repurposing as a link prediction task. KGE models allow for the enrichment of existing knowledge by incorporating multimodal information from various sources. In this study, we constructed RDKG-115, a rare disease knowledge graph involving 115 RDs, composed of 35,643 entities, 25 relations, and 5,539,839 refined triplets, based on 372,384 high-quality literature and 4 biomedical datasets: DRKG, Pathway Commons, PharmKG, and PMapp. Subsequently, we developed a trimodal KGE model containing structure, category, and description embeddings using reverse-hyperplane projection. We utilized this model to infer 4,199 reliable new inferred triplets from RDKG-115. Finally, we calculated potential drugs and small molecules for each of the 115 RDs, taking multiple sclerosis as a case study. This study provides a paradigm for large-scale screening of drug repurposing and discovery for RDs, which will speed up the drug development process and ultimately benefit patients with RDs.    

## Files
### Annotation/
**E_dict.json** : json file of standard entity set  

### Model/  
**Models.py** : TransE, TransH, ConvKB, RotatE structure    
**run_kge.py** : Running and configuration file         
**train_kge.py** : Class of processing and tool functions for Knowledge Graph Embedding   
#### C/
**C_dict.data** : Dict of entity category annotation  
**get_c_dict.py** : Run it to get C_dict.data  
#### D/
**BERT.py** : Structure of BERT  
**DTModel.py** : Structure for training description table    
**optimization.py** : Training optimization of BERT     
**run_d_Table.py** : Run it to get D_table.data  
**tokenization.py** : Tokenization function of BERT   
**train_d_Table.py** : Class of processing and tool functions for D_table   
#### KG/
**entity.csv** : Entities of RDKG-115  
**relation.csv** : Relations of RDKG-115  
**train.csv** : Train set of RDKG-115 (98%)    
**dev.csv** : Dev set of RDKG-115 (1%)  
**test.csv** : Test set of RDKG-115 (1%)  
#### Pretrained BERT/ 
##### bert/ : Self download from https://huggingface.co/bert-base-uncased/tree/main  
##### biobert/ : Self download from https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main  
##### pubmedbert/ : Self download from https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/tree/main  
##### scibert/ : Self download from https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main  

### RDKG-115/
**entity_name_map.xlsx** : Entity key-name mapping file  
**RDKG-115.csv** : Raw RDKG-115   
**RDKG-115 (plus inferred).csv** : RDKG-115 merge with reliable new inferred knowledges  

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)  
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network.](https://arxiv.org/abs/1712.02121.pdf)  
(4) **RotatE**: [ROTATE: KNOWLEDGE GRAPH EMBEDDING BY RELATIONAL ROTATION IN COMPLEX SPACE](https://arxiv.org/pdf/1902.10197.pdf)   
(5) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)    
(6) **BioBERT**: [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746.pdf)  
(7) **PubMedBERT**: [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://arxiv.org/pdf/2007.15779.pdf)  
(8) **SCIBERT**: [SCIBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676.pdf)  

## Version
(1) python 3.6.2  
(2) torch 1.9.1+cu111  
(3) numpy 1.19.2

## Operating Instructions
(1) Run get_c_dict.py to get C_dict.data in **Model/C/** (Already in the folder, you can not run)    
```
python get_c_dict.py   
```

(2) Run run_d_table.py to get D_table.data in **Model/D/**     
```
python run_d_table.py --model pubmedbert --len_d 150 --l_r 1e-5 --batch_size 8 --epoches 5 --earlystop 1   
```

(3) Run run_kge.py to train TransE, TransH, ConvKB and RotatE in **Model/**
#### 4 Configurations Interpretation   
lanta_c == 0 and lanta_d == 0 : S  
lanta_c != 0 and lanta_d == 0 : S + C  
lanta_c == 0 and lanta_d != 0 : S + D  
lanta_c != 0 and lanta_d != 0 : S + C + D  

#### Recommended Parameters for 4 models and 4 Configurations (dim of 128)   
**TransE**
```
python run_kge.py --model TransE --margin 1.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 800
python run_kge.py --model TransE --margin 1.0 --lanta_c 0.5 --lanta_d 0.0 --l_r 5e-4 --epoches 200
python run_kge.py --model TransE --margin 1.0 --lanta_c 0.0 --lanta_d 0.5 --l_r 5e-4 --epoches 200
python run_kge.py --model TransE --margin 1.0 --lanta_c 0.1 --lanta_d 0.5 --l_r 5e-4 --epoches 200
```
**TransH**
```
python run_kge.py --model TransH --margin 1.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 800
python run_kge.py --model TransH --margin 1.0 --lanta_c 0.3 --lanta_d 0.0 --l_r 5e-4 --epoches 200
python run_kge.py --model TransH --margin 1.0 --lanta_c 0.0 --lanta_d 0.5 --l_r 5e-4 --epoches 200
python run_kge.py --model TransH --margin 1.0 --lanta_c 0.3 --lanta_d 0.5 --l_r 5e-4 --epoches 200
```
**ConvKB**
```
python run_kge.py --model ConvKB --margin 2.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 800
python run_kge.py --model ConvKB --margin 2.0 --lanta_c 0.3 --lanta_d 0.0 --l_r 5e-4 --epoches 200
python run_kge.py --model ConvKB --margin 2.0 --lanta_c 0.0 --lanta_d 0.1 --l_r 5e-4 --epoches 200
python run_kge.py --model ConvKB --margin 2.0 --lanta_c 0.3 --lanta_d 0.5 --l_r 5e-4 --epoches 200
```
**RotatE**
```
python run_kge.py --model RotatE --margin 3.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 800
python run_kge.py --model RotatE --margin 3.0 --lanta_c 0.3 --lanta_d 0.0 --l_r 5e-4 --epoches 200
python run_kge.py --model RotatE --margin 3.0 --lanta_c 0.0 --lanta_d 0.3 --l_r 5e-4 --epoches 200
python run_kge.py --model RotatE --margin 3.0 --lanta_c 0.1 --lanta_d 0.5 --l_r 5e-4 --epoches 200
```
