import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
from pprint import pprint
from typing import List
from datautils import get_operator_class

import warnings
warnings.filterwarnings('ignore')

dataset = pd.DataFrame(columns = ['uid', 'order', 'question', 'answer', 'label'])


mode = 'train'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
f = open(f"{mode}_log.txt", "w")


count, total = 0, 0

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    # text = [para['text'] for para in paragraphs]

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        operator = get_operator_class(
            ques['derivation'], 
            ques['answer_type'],
            ques['facts'],
            ques['answer'],
            ques['mapping']
            )
        if operator == None:
            count += 1 
            f.write(f"UID: {uid}\n")
            f.write(f"ORDER: {ques_order}\n")
            f.write(f"DERIVATION: {ques['derivation']}\n")
            f.write(f"ANSWER_TYPE: {ques['answer_type']}\n")
            f.write(f"FACTS: {ques['facts']}\n")
            f.write(f"ANSWER: {ques['answer']}\n")
            f.write(f"MAPPING: {ques['mapping']}\n")
            f.write(f"QUESTION: {ques['question']}\n")
            f.write('=' * 60 + '\n')
            operator = 10
            # print(ques)
        # else: 
        row = {
            'uid':uid,
            'order': ques_order,
            'question':ques['question'],
            'answer': ques['answer'], 
            'label' : operator    
        }
        dataset = dataset.append(row, ignore_index = True)

        total += 1 

print('count', count, 'total', total )



dataset.to_csv(f'dataset_tagop/{mode}.csv', index = False)
f.close()


# train count 385 total 13215
# dev   count 60 total 1668