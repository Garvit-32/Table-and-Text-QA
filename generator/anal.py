import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

tmp = []

mode = 'dev'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
count, total = 0, 0

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    table = data[i]['table']
    table_data = table['table']
    questions = data[i]['questions']
    
    for ques in questions:
        tmp.append(ques['answer_type'])

print(np.unique(tmp))
        