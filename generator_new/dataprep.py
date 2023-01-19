import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
import os
import re

import warnings
warnings.filterwarnings('ignore')


dataset = pd.DataFrame(columns = ['uid', 'order','type','text', 'answer'])
# dataset = pd.DataFrame(columns = ['uid', 'order','type', "question",  'text', 'answer'])


mode = 'train'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
# f = open(f"{mode}_log.txt", "w")

count, total = 0, 0

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    table = data[i]['table']
    table_data = table['table']
    questions = data[i]['questions']
    

    para_text = [para['text'].strip() for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for i in table_data:
        for j in i:
            if len(j):
                table_tokens.append(j.strip())
    
    table_text = ' '.join(table_tokens)

    text = " </s> " + table_text + " </s> " + para_text  

    # text = table_text + " " + para_text

    for ques in questions:
                
        ques_order = ques['order']
        answers = ques['answer']
        scale = ques['scale']
        answer_type = ques['answer_type']
        


        if answer_type in ['arithmetic']: 
            answer = str(ques['derivation'].strip())       
            answer = re.sub('[[]', '(', answer)       
            answer = re.sub('[]]', ')', answer)       
            answer = re.sub(' ', '', answer)       

            if len(scale):
                answer += f' {scale}'
        
        else:
            continue
        # if answer_type != 'count': continue

            
        #     # answer += ' <a>'
        
        # elif answer_type == 'count':
        #     answer = ' ^ '.join(ques['facts'])
        
        #     if len(scale):
        #         answer += f' {scale}'
            
        # else:
        #     if isinstance(answers,(float, int)):
        #         answer = str(answers)
            
        #     elif isinstance(answers,str):
        #         answer = answers
        #     elif isinstance(answers,list):
        #         answer = list(map(str, answers))
        #         answer = ' '.join(answers)
        #     if len(scale):
        #         answer += f' {scale}'
            
        tmp = ques['question'] + text

        row = {
            'uid':uid,
            'order': ques_order,
            'type': ques['answer_type'],
            'text': tmp,
            'answer': answer, 
            
        }

        dataset = dataset.append(row, ignore_index = True)




dataset.to_csv(f'dataset_tagop/{mode}_arithmetic.csv', index = False)

# 68bafc82-b795-4c7b-9506-e901223c3526,3


# c6df9f0a-4811-48ab-a453-0cb6a7baa35c,4

# # train count 385 total 13215
# # dev   count 60 total 1668