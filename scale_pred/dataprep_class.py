import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

dataset = pd.DataFrame(columns = ['uid', 'order', 'scale', 'text', 'label'])


mode = 'train'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))


data_dict = {"": 0, "thousand": 1, "million": 2,  "billion": 3, "percent": 4}

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    table = data[i]['table']
    table_data = table['table']
    questions = data[i]['questions']
    

    para_text = [para['text'].strip() for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for a in table_data:
        for b in a:
            if len(b):
                table_tokens.append(b.strip())
    
    table_text = ' '.join(table_tokens)

    text = " </s> " + table_text + " </s> " + para_text  
    # text = " [SEP] " + table_text + " [SEP] " + para_text  

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        operator = data_dict[ques['scale']]
        tmp = ques['question'] + text
        
        # else: 
        row = {
            'uid':uid,
            'order': ques_order,
            'scale': ques['scale'], 
            'text': tmp, 
            'label' : operator    
        }
        dataset = dataset.append(row, ignore_index = True)

dataset.to_csv(f'dataset_tagop/{mode}_roberta.csv', index = False)


# train count 385 total 13215
# dev   count 60 total 1668