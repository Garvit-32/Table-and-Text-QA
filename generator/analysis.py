import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
import os


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

    para_text = [para['text'].strip() for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for i in table_data:
        for j in i:
            if len(j):
                table_tokens.append(j.strip())
    
    table_text = ' '.join(table_tokens)

    text = " </s> " + table_text + " </s> " + para_text



    for ques in questions:
        

        ques_order = ques['order']
        answers = ques['answer']
        scale = ques['scale'].strip()
        

        if isinstance(answers,list) and scale != '':
            # count += 1
            print(uid, ques_order)
        
        

        # if scale == '':
        #     total += 1

print(count , total)

        
        # if isinstance(answers,(float, int)):
        #     answer = str(answers)
        
        # elif isinstance(answers,str):
        #     answer = answers

        # elif isinstance(answers,list):
        #     # answers.sort()
        #     answer = list(map(str, answers))
        #     # answer = ' '.join(answers)
        

        # if len(scale):
        #     answer += f' {scale}'


        # elif isinstance(answers,list):
        #     # answers.sort()
        #     answers = list(map(str, answers))
        #     answer = ' '.join(answers)
            
        # tmp = ques['question'] + text

#         row = {
#             'uid':uid,
#             'order': ques_order,
#             'text': tmp,
#             'answer': answer, 
        
#         }

#         dataset = dataset.append(row, ignore_index = True)




# dataset.to_csv(f'dataset_tagop/{mode}_exact_w_scale.csv', index = False)


# # train count 385 total 13215
# # dev   count 60 total 1668