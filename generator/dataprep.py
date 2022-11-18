import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
from pprint import pprint
from typing import List


import warnings
warnings.filterwarnings('ignore')


dataset = pd.DataFrame(columns = ['uid', 'order', 'text', 'answer'])


mode = 'dev'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
# f = open(f"{mode}_log.txt", "w")


count, total = 0, 0

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    table = data[i]['table']
    table_data = table['table']
    questions = data[i]['questions']

    para_text = [para['text'] for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for i in table_data:
        for j in i:
            if len(j):
                table_tokens.append(j)
    
    table_text = ' '.join(table_tokens)

    text = " </s> " + table_text + " </s> " + para_text



    for ques in questions:
        

        ques_order = ques['order']
        answer = str(ques['answer'])
        
        text = ques['question'] + text
        
        row = {
            'uid':uid,
            'order': ques_order,
            'text': text,
            'answer': answer, 
        
        }
        dataset = dataset.append(row, ignore_index = True)




dataset.to_csv(f'dataset_tagop/{mode}.csv', index = False)


# # train count 385 total 13215
# # dev   count 60 total 1668