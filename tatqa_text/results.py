import ast
import torch 
import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import json
from utils import *


import evaluate
metric = evaluate.load("squad_v2")


mode = 'train'
dataset_path = 'dataset_tagop'
table_csv_path = 'dataset_tagop/dev'
model_checkpoint = "bert-large-bio-bs-32"
# model_checkpoint = "bert-large-num-bs-16"
# model_checkpoint = "bert-base-bs-32"
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))


token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="average", device = 0
)

def postprocess(uid, order, output, answer):
    text  = []

    # if len(output):
    #     if output[-1]['word'] ==  '.':
    #         output.pop()

    tmp = ''
    prev_end = 0

    for idx, i in enumerate(output):
        
        start = i['start']
        end = i['end']
        word = i['word']
        
        if prev_end + 1 == start:
            tmp += ' '

        elif prev_end != start and idx != 0:
            text.append(tmp)
            tmp = '' 

        tmp += word
        if idx == len(output) - 1:
            text.append(tmp)
        
        prev_end = end
    
    uid_order = uid + '_' + str(order)
    pred_dict = {'id' : uid_order}
    
    if len(text):
        text = list(map(str, text))
        text.sort()
        text = ' '.join(text)
        pred_dict['no_answer_probability'] = 0.
    else:
        text = ''
        pred_dict['no_answer_probability'] = 1.


    pred_dict['prediction_text'] = text
    

    text = list(map(str, text))
    text = text.sort()
    answer = ' '.join(answer)


    gt_dict = {'id' : uid_order}
    gt_dict['answers'] = {"text": [answer], 'answer_start':[0]}
    
    result = metric.compute(predictions=[pred_dict], references=[gt_dict])
    return result['exact'], result['f1']

exact,f1 = [], []
   

for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    paras = ' '.join([para['text'] for para in paragraphs])
    # paras = paras.split(' ')
    # new_paras = []
    # for p in paras:
    #     num = to_number(p)
    #     if num is not None:
    #         new_paras.append(str(num))
    #     else:
    #         new_paras.append(p)
    
    # new_paras = ' '.join(new_paras)
    

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        mappings = ques['mapping']
      
        question = ques['question']

        if ques['answer_type'] != 'arithmetic' and ques['answer_from'] != 'table-text':

            text = question + ' [SEP] ' + paras
            # text = question + ' [SEP] ' + paras

            output = token_classifier(text)
            if ques['answer_from'] == 'table':
                ques['answer'] = []

            arg1, arg2 = postprocess(uid, ques_order, output ,ques['answer'])
            # print(arg1, arg2)
            # if arg1 + arg2 == 0.0:
                # print(uid, ques_order)

            exact.append(arg1)
            f1.append(arg2)

print(len(exact), len(f1))
print(np.mean(exact))
print(np.mean(f1))

# 687d09ee-4aa8-402a-ae27-96ce063115a5 3
# eb122e2e-12b4-4e59-a018-7045192b021d 1
# eb122e2e-12b4-4e59-a018-7045192b021d 2



# 52.31481481481482
# 72.37284801924456


# 55.401234567901234
# 76.32378104053979

# tweaks
# 61.26543209876543
# 81.52669589300007


# bert-large-bio-bs-16
# 69.08456167571761
# 85.44454493738296

# bert-large-bio-bs-32
# 68.79363847944143
# 85.40983374630115

# [{'entity_group': 'ANS', 'score': 0.99846727, 'word': '268', 'start': 528, 'end': 531}, {'entity_group': 'ANS', 'score': 0.99950314, 'word': ',', 'start': 531, 'end': 532}, {'entity_group': 'ANS', 'score': 0.9994068, 'word': '000', 'start': 532, 'end': 535}]
