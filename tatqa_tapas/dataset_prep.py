
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import json 
import pandas as pd
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')

mode = 'train'
path = 'dataset_tagop_table'

os.makedirs(f'{path}/{mode}', exist_ok = True)
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
dataset = pd.DataFrame(columns = ['uid', 'position', 'question','answer_cord','answer'])


flag_true = False
for i in tqdm(range(len(data))):
    table = data[i]['table']
    uid = table['uid']
    flag = False

    table_data = table['table']

    table = pd.DataFrame(table_data, columns = [''] * len(table_data[0])).astype(str)
    table.to_csv(f'{path}/{mode}/{uid}.csv', index = False)

    questions = data[i]['questions']
    for ques in questions:
        ques_id = ques['order']-1
        question = ques['question']
        if isinstance(ques['answer'],list):
            answer = ques['answer'] 
        else:
            answer = [str(ques['answer'])]

        if 'table' in ques['mapping'].keys() :
            if len(ques['mapping']['table']) == 0:
                answer_coord =[]
                answer = []
            else:
                answer_coord = [tuple(j) for j in ques['mapping']['table']] 
        else: 
            
            answer_coord = []
            answer = []
        
        # if flag: continue

        row = {
            'uid':uid,
            'position':ques_id,
            'question':question,
            'answer_cord':answer_coord,
            'answer':answer
        }
        dataset = dataset.append(row, ignore_index = True)

    #     if flag_true: break
    
    # if flag_true: break
    
dataset.to_csv(f'{path}/{mode}.csv', index = False)
