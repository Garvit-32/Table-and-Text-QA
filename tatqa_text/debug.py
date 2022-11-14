import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
from pprint import pprint
from typing import List

mode = 'train'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))

both = 0
table = 0
para = 0

count = 0

for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    text = [para['text'] for para in paragraphs]
    questions = data[i]['questions']
    for ques in questions:
        order = ques['order']
        mapping = ques['mapping'].keys()
        answer_type = ques['answer_type']
        
        if 'arithmetic' == answer_type:
            count += 1

            if 'table' in mapping and 'paragraph' in mapping:
                print(uid, order)
                both += 1

            if 'table' in mapping and 'paragraph' not in mapping:
                table += 1
                    
            if 'table' not in mapping and 'paragraph' in mapping:
                para += 1


print('table', table)
print('para', para)
print('both', both)
print('total', count)
        
        
        
# table 9620
# para 3125
# both 470
# total 13215


# table 5337
# para 116
# both 90
# total 5543
